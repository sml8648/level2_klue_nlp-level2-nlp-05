import re
from datetime import datetime
from pydoc import locate

import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

import transformers
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import AutoModelForSequenceClassification

import data_loaders.data_loader as dataloader
from data_loaders.data_loader import MyDataCollatorWithPadding
import utils.util as utils

import mlflow
import mlflow.sklearn
from azureml.core import Workspace

from omegaconf import OmegaConf


def start_mlflow(experiment_name):
    # Enter details of your AzureML workspace
    subscription_id = "0275dc6c-996d-42d1-8263-8f7b4e81f271"
    resource_group = "basburger"
    workspace_name = "basburger"
    ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    mlflow.set_experiment(experiment_name)
    # Start the run
    mlflow.start_run()


def train(conf):
    # 실행 시간을 기록합니다.
    now = datetime.now()
    train_start_time = now.strftime("%d-%H-%M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # add special token in rbert model
    if conf.data.dataloader == "typed_entity_marker_emask":
        special_tokens_dict = {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>", "<e3>", "</e3>", "<e4>", "</e4>"]}
        tokenizer.add_special_tokens(special_tokens_dict)

    data_collator = MyDataCollatorWithPadding(tokenizer=tokenizer)

    # mlflow 실험명으로 들어갈 이름을 설정합니다.
    experiment_name = model_name + "_" + conf.model.model_class_name + "_bs" + str(conf.train.batch_size) + "_ep" + str(conf.train.max_epoch) + "_lr" + str(conf.train.learning_rate)
    start_mlflow(experiment_name)  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.

    # load dataset
    RE_train_dataset = dataloader.load_dataset(tokenizer, conf.path.train_path, conf)
    RE_dev_dataset = dataloader.load_dataset(tokenizer, conf.path.dev_path, conf)
    RE_test_dataset = dataloader.load_dataset(tokenizer, conf.path.test_path, conf)
    RE_predict_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.predict_path, conf)

    if conf.train.continue_train:
        model_class = locate(f"model.{conf.model.model_type}.{conf.model.model_class_name}")
        model = model_class(conf, len(tokenizer))
        checkpoint = torch.load(conf.path.load_model_path)
        model.load_state_dict(checkpoint)
    # TAPT로 학습된 모델 로드
    elif conf.model.use_tapt_model:
        model = AutoModelForSequenceClassification.from_pretrained(conf.path.load_pretrained_model_path, num_labels=30)
    else:
        model_class = locate(f"model.{conf.model.model_type}.{conf.model.model_class_name}")
        model = model_class(conf, len(tokenizer))

    model.parameters
    model.to(device)
    # 다른 옵티마이저를 사용하고 싶으신 경우 이 부분을 바꿔주세요.
    optimizer = transformers.AdamW(model.parameters(), lr=conf.train.learning_rate)

    # 이등변 삼각형 형태로 lr이 서서히 증가했다가 감소하는 스케줄러입니다.
    # 첫시작 lr: learning_rate/div_factor, 마지막 lr: 첫시작 lr/final_div_factor
    # 학습과정 step수를 계산해 스케줄러에 입력해줍니다. -> steps_per_epoch * epochs / 2 지점 기준으로 lr가 상승했다가 감소
    steps_per_epoch = len(RE_train_dataset) // conf.train.batch_size + 1 if len(RE_train_dataset) % conf.train.batch_size != 0 else len(RE_train_dataset) // conf.train.batch_size
    scheduler = OneCycleLR(
        optimizer,
        max_lr=conf.train.learning_rate,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.5,
        epochs=conf.train.max_epoch,
        anneal_strategy="linear",
        div_factor=1e100,
        final_div_factor=1,
    )

    training_args = TrainingArguments(
        # step_saved_model 폴더에 저장됩니다.
        output_dir=f"./step_saved_model/{re.sub('/', '-', model_name)}/{train_start_time}",
        save_total_limit=conf.utils.top_k,  # save_steps에서 저장할 모델의 최대 개수
        save_steps=conf.train.save_steps,  # 이 step마다 eval_steps에서 계산한 값을 기준으로 모델을 저장합니다.
        num_train_epochs=conf.train.max_epoch,  # 학습 에포크 수
        learning_rate=conf.train.learning_rate,  # learning_rate
        per_device_train_batch_size=conf.train.batch_size,  # train batch size
        per_device_eval_batch_size=conf.train.batch_size,  # valid batch size
        logging_dir="./logs",
        logging_steps=conf.train.logging_steps,  # 해당 스탭마다 loss, lr, epoch가 cmd에 출력됩니다.
        evaluation_strategy="steps",
        eval_steps=conf.train.eval_steps,  # 해당 스탭마다 valid set을 이용해서 모델을 평가합니다. 이 값을 기준으로 save_steps 모델이 저장됩니다.
        load_best_model_at_end=True,
        metric_for_best_model=conf.utils.monitor,  # 평가 기준으로 할 loss값을 설정합니다.
    )
    trainer = Trainer(
        model=model,
        args=training_args,  # 위에서 설정한 training_args를 가져옵니다.
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=utils.compute_metrics,  # utils에 있는 평가 매트릭을 가져옵니다.
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=conf.utils.patience)],
    )

    trainer.train()
    # train 과정에서 가장 평가 점수가 좋은 모델을 저장합니다.
    # best_model 폴더에 저장됩니다.
    trainer.save_model(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}")

    mlflow.end_run()  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.
    model.eval()
    metrics = trainer.evaluate(RE_test_dataset)
    print("Training is complete!")
    print("==================== Test metric score ====================")
    print("eval loss: ", metrics["eval_loss"])
    print("eval auprc: ", metrics["eval_auprc"])
    print("eval micro f1 score: ", metrics["eval_micro f1 score"])

    # best_model 저장할 때 사용했던 config파일도 같이 저장합니다.
    with open(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/config.yaml", "w+") as fp:
        OmegaConf.save(config=conf, f=fp.name)

    # best_model 로드
    load_model_path = f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/pytorch_model.bin"
    checkpoint = torch.load(load_model_path)
    model_class = locate(f"model.{conf.model.model_type}.{conf.model.model_class_name}")
    model = model_class(conf, len(tokenizer))
    model.load_state_dict(checkpoint)

    model.parameters
    model.to(device)
    model.eval()

    test_args = TrainingArguments(output_dir="./prediction", do_train=False, do_predict=True, per_device_eval_batch_size=16, dataloader_drop_last=False)
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics, data_collator=data_collator)

    # Test 점수 확인
    predict_dev = True  # dev set에 대한 prediction 결과값 구하기 (output분석)
    predict_submit = True  # dev set은 evaluation만 하고 submit할 결과값 구하기
    if predict_dev:
        outputs = trainer.predict(RE_test_dataset)
        logits = torch.FloatTensor(outputs.predictions)
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        result = torch.argmax(logits, axis=-1).detach().cpu().numpy()

        pred_answer = result.tolist()
        pred_answer = utils.num_to_label(pred_answer)
        output_prob = prob.tolist()

        output = pd.read_csv("./dataset/test/test.csv")
        output["pred_label"] = pred_answer
        output["probs"] = output_prob

        output.to_csv(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/dev_submission_{train_start_time}.csv", index=False)
        output.to_csv(f"./prediction/dev_submission_{train_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    if predict_submit:
        outputs1 = trainer.predict(RE_predict_dataset)
        logits1 = torch.FloatTensor(outputs1.predictions)
        prob1 = F.softmax(logits1, dim=-1).detach().cpu().numpy()
        result1 = torch.argmax(logits1, axis=-1).detach().cpu().numpy()

        pred_answer1 = result1.tolist()
        pred_answer1 = utils.num_to_label(pred_answer1)
        output_prob1 = prob1.tolist()

        output1 = pd.read_csv("./prediction/sample_submission.csv")
        output1["pred_label"] = pred_answer1
        output1["probs"] = output_prob1

        output1.to_csv(f"./prediction/submission_{train_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
        output1.to_csv(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/submission_{train_start_time}.csv", index=False)
