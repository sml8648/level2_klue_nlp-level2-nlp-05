from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from transformers import AutoConfig
import model.model as model_arch

import data_loaders.data_loader as dataloader
import utils.util as utils

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if conf.data.tem: #typed entity token에 쓰이는 스페셜 토큰
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>', '<e3>', '</e3>', '<e4>', '</e4>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    # .bin을 가져옵니다.
    load_model_path = conf.path.load_model_path
    checkpoint = torch.load(load_model_path)

    # 모델 구조를 가져옵니다.
    if conf.model.exp_name == 'Model':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=30)
        model.resize_token_embeddings(len(tokenizer))
    elif conf.model.exp_name == 'CustomRBERT':    #RBERT
        model_config = AutoConfig.from_pretrained(model_name)
        model = model_arch.CustomRBERT(model_config, conf, len(tokenizer))

    # 모델 구조 위에 checkpoint를 덮어씌웁니다.
    # 모델 구조와 checkpoint에 저장되어 있는 파라미터 구조가 다른 경우 에러가 발생합니다.
    # 에러를 무시하면서 파라미터를 입력하고 싶은 경우엔 strich=False를 인자로 설정합니다.
    model.load_state_dict(checkpoint, strict=False)
    model.parameters
    model.to(device)

    ## load predict datset
    RE_predict_dataset, predict_id = dataloader.load_predict_dataset(tokenizer, conf)

    # arguments for Trainer
    # predict data를 padding없이 입력하기 위해 batch_size를 1로 입력합니다.
    # batch_size를 키우고 싶은 경우엔 trainer에 collator를 추가해서 실행시켜주세요.
    test_args = TrainingArguments(output_dir="./prediction", do_train=False, do_predict=True, per_device_eval_batch_size=1, dataloader_drop_last=False)

    # init trainer
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics)
    outputs = trainer.predict(RE_predict_dataset.pair_dataset)
    logits = torch.tensor(outputs.predictions)
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    result = torch.argmax(logits, axis=-1).detach().cpu().numpy()

    pred_answer = result.tolist()
    pred_answer = utils.num_to_label(pred_answer)
    output_prob = prob.tolist()

    output = pd.DataFrame(
        {
            "id": predict_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv(f"./prediction/submission_{inference_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("==================== Inference finish! ====================")
