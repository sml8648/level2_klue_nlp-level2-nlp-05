from transformers import AutoTokenizer, Trainer, TrainingArguments

import data_loaders.data_loader as dataloader
import utils.util as utils
import model.model as model_arch

import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
from datetime import datetime


def inference(conf):
    # 실행 시간을 기록합니다.
    now = datetime.now()
    inference_start_time = now.strftime("%d-%H-%M")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fase=False)

    # 이후 토큰을 추가하는 경우 이 부분에 추가해주세요.
    # tokenizer.add_special_tokens()
    # tokenizer.add_tokens()

    # .bin을 가져옵니다.
    load_model_path = conf.path.load_model_path
    checkpoint = torch.load(load_model_path)
    # 모델 구조를 가져옵니다. 반드시 학습할 때 사용했던 동일한 모델 클래스를 사용해야 합니다.
    model = model_arch.Model(conf, len(tokenizer))
    # 모델 구조 위에 checkpoint(파라미터)를 덮어씌웁니다.
    # 모델 구조와 checkpoint에 저장되어 있는 파라미터 구조가 다른 경우 에러가 발생합니다.
    model.load_state_dict(checkpoint)
    model.parameters
    model.to(device)
    model.eval()

    ## load predict datset
    predict_dataset_dir = conf.path.predict_path
    RE_predict_dataset = dataloader.load_predict_dataset(tokenizer, predict_dataset_dir)

    # arguments for Trainer
    # predict data를 padding없이 입력하기 위해 batch_size를 1로 입력합니다.
    # batch_size를 키우고 싶은 경우엔 trainer에 collator를 추가해서 실행시켜주세요.
    test_args = TrainingArguments(output_dir="./prediction", do_train=False, do_predict=True, per_device_eval_batch_size=1, dataloader_drop_last=False)

    # init trainer
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics)
    outputs = trainer.predict(RE_predict_dataset)
    logits = torch.tensor(outputs.predictions)
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    result = torch.argmax(logits, axis=-1).detach().cpu().numpy()

    pred_answer = result.tolist()
    pred_answer = utils.num_to_label(pred_answer)
    output_prob = prob.tolist()

    output = pd.read_csv("./prediction/sample_submission.csv")
    output["pred_label"] = pred_answer
    output["probs"] = output_prob

    output.to_csv(f"./prediction/submission_{inference_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("==================== Inference finish! ====================")
