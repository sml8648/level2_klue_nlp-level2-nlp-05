import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

# https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html
import data_loaders.data_loader as dataloader
import trainer.trainer as CustomTrainer
import utils.util as utils
import transformers
from torch.optim.lr_scheduler import OneCycleLR

import model.model as model_arch
import model.modeling_roberta as roberta_arch
from transformers import DataCollatorWithPadding

# https://huggingface.co/course/chapter3/4

import mlflow
import mlflow.sklearn

from ray import tune
from azureml.core import Workspace
import argparse

from omegaconf import OmegaConf
from datetime import datetime
import re
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
from pydoc import locate

class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len = 0
        for i in features:
            if len(i['input_ids']) > max_len : max_len = len(i['input_ids'])

        batch = defaultdict(list)
        for item in features:
            for k in item:
                if('label' not in k):
                    padding_len = max_len - item[k].size(0)
                    if(k == 'input_ids'):
                        item[k] = torch.cat((item[k], torch.tensor([self.tokenizer.pad_token_id]*padding_len)), dim=0)
                    else:
                        item[k] = torch.cat((item[k], torch.tensor([0]*padding_len)), dim=0)
                batch[k].append(item[k])
                
        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
            batch[k] = batch[k].to(torch.long)
        return batch



def start_mlflow(experiment_name):
    # Enter details of your AzureML workspace
    subscription_id = "0275dc6c-996d-42d1-8263-8f7b4e81f271"
    resource_group = "basburger"
    workspace_name = "basburger"
    ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-log-view-metrics?tabs=interactive
    mlflow.set_experiment(experiment_name)
    # Start the run
    mlflow_run = mlflow.start_run()


def train(args, conf):
    now = datetime.now()
    train_start_time = now.strftime("%d-%H-%M")

    # huggingface-cli login  #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

    model_name = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    data_collator = MyDataCollatorWithPadding(tokenizer=tokenizer)

    new_token_count = 0
    if conf.data.tem == 2: #typed entity token에 쓰이는 스페셜 토큰
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>', '<e3>', '</e3>', '<e4>', '</e4>']}
        tokenizer.add_special_tokens(special_tokens_dict)
    # new_token_count += tokenizer.add_special_tokens()
    # new_token_count += tokenizer.add_tokens()
    new_vocab_size = tokenizer.vocab_size + new_token_count
    print(new_vocab_size, len(tokenizer))

    experiment_name = model_name + "_bs" + str(conf.train.batch_size) + "_ep" + str(conf.train.max_epoch) + "_lr" + str(conf.train.learning_rate)
    start_mlflow(experiment_name)

    # load dataset
    RE_train_dataset = dataloader.load_dataset(tokenizer, conf.path.train_path,conf)
    RE_dev_dataset = dataloader.load_dataset(tokenizer, conf.path.dev_path,conf)
    RE_test_dataset = dataloader.load_dataset(tokenizer, conf.path.test_path,conf)
    RE_predict_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.predict_path,conf)

    # 모델을 로드합니다. 커스텀 모델을 사용하시는 경우 이 부분을 바꿔주세요.
    continue_train=False
    if continue_train:    
        model_config = AutoConfig.from_pretrained(model_name)
        model = model_arch.CustomRBERT(model_config, conf, len(tokenizer))
        checkpoint = torch.load(conf.path.load_model_path)
        model.load_state_dict(checkpoint)
    elif conf.model.model_class_name == 'TAPT' :
        model = AutoModelForSequenceClassification.from_pretrained(
        conf.path.load_pretrained_model_path, num_labels=30
        )
    else:
        model_class = locate(f'model.model.{conf.model.model_class_name}')
        if model_class == None :
             model_class = locate(f'model.modeling_roberta.{conf.model.model_class_name}') # for modeling_roberta
        model = model_class(conf, len(tokenizer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.parameters
    model.to(device)
    
    def ray_hp_space(trial):
        return {
            "learning_rate": tune.loguniform(8e-6, 6e-5),
            "per_device_train_batch_size": tune.choice([16]),
        }

    def model_init(trial):
        model_class = locate(f'model.model.{conf.model.model_class_name}')
        if model_class == None :
            model_class = locate(f'model.modeling_roberta.{conf.model.model_class_name}') # for modeling_roberta
        model = model_class(conf, len(tokenizer))
        return model

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        #hub_model_id="jstep750/basburger",
        output_dir=f"./step_saved_model/{re.sub('/', '-', model_name)}/{train_start_time}",  # output directory
        save_total_limit=1,  # number of total save model.
        save_steps=914,  # model saving step.
        num_train_epochs=conf.train.max_epoch,  # total number of training epochs
        learning_rate=conf.train.learning_rate,  # learning_rate
        per_device_train_batch_size=conf.train.batch_size,  # batch size per device during training
        per_device_eval_batch_size=conf.train.batch_size,  # batch size for evaluation
        # weight_decay=0.01,               # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=500,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=914,  # evaluation step.
        load_best_model_at_end=True,
        #push_to_hub=False,
        metric_for_best_model="micro f1 score",
    )

    # for hyper parameter search
    custom_trainer = CustomTrainer.CustomTrainer(
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=utils.compute_metrics,  # define metrics function
        data_collator=data_collator,
        model_init=model_init,
        # optimizers=(optimizer, scheduler), # Error : `model_init` is incompatible with `optimizers`
        callbacks=[EarlyStoppingCallback(early_stopping_patience=conf.utils.patience)],
        model=model,  # 🤗 for Transformers model parameter
        conf=conf,
    )

    best_run = custom_trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=ray_hp_space,
        n_trials=2,
    )

    print(best_run)
    print("Before:", custom_trainer.args)
    # 참고 예정
    # best_run으로 받아온 best hyperparameter로 재학습
    # https://github.com/huggingface/setfit/blob/ebee18ceaecb4414482e0a6b92c97f3f99309d56/scripts/transformers/run_fewshot.py
    for key, value in best_run.hyperparameters.items():
        setattr(custom_trainer.args, key, value)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_train_dataset,  # evaluation dataset
        compute_metrics=utils.compute_metrics,  # define metrics function
        data_collator=data_collator,
        # optimizers=optimizers,
        # model_init=model_init,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("After:", trainer.args)

    # train model
    trainer.train()
    # trainer.push_to_hub() # Error! DENIED update refs/heads/dev: forbidden
    trainer.save_model(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}")
    
    mlflow.end_run()  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.
    # trainer.push_to_hub()  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.
    model.eval()
    metrics = trainer.evaluate(RE_test_dataset)
    print("Training is complete!")
    print("==================== Test metric score ====================")
    print("eval loss: ", metrics["eval_loss"])
    print("eval auprc: ", metrics["eval_auprc"])
    print("eval micro f1 score: ", metrics["eval_micro f1 score"])

    
    # best_model 저장할 때 사용했던 config파일도 같이 저장합니다.
    if not os.path.exists(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}"):
        os.makedirs(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}")
    with open(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/config.yaml", "w+") as fp:
        OmegaConf.save(config=conf, f=fp.name)

    test_args = TrainingArguments(output_dir="./prediction", do_train=False, do_predict=True, per_device_eval_batch_size=16, dataloader_drop_last=False)
    # init trainer
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics, data_collator=data_collator)

    # Test 점수 확인
    predict_dev = True  # dev set에 대한 prediction 결과값 구하기 (output분석)
    predict_submit = True # dev set은 evaluation만 하고 submit할 결과값 구하기
    if(predict_dev):
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
    if(predict_submit):
        metrics = trainer.evaluate(RE_test_dataset)
        print("Training is complete!")
        print("==================== Test metric score ====================")
        print("eval loss: ", metrics["eval_loss"])
        print("eval auprc: ", metrics["eval_auprc"])
        print("eval micro f1 score: ", metrics["eval_micro f1 score"])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="roberta_entity_config")
    parser.add_argument("--shuffle", default=True)
    # parser.add_argument('--optimizer', default='AdamW')

    parser.add_argument("--preprocessing", default=False)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()

    conf = OmegaConf.load(f"./config/{args.config}.yaml")
    # check hyperparameter arguments
    print(args)
    train(args, conf)