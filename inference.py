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
from transformers import DataCollatorWithPadding
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


def inference(conf):
    # 실행 시간을 기록합니다.
    now = datetime.now()
    inference_start_time = now.strftime("%d-%H-%M")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = MyDataCollatorWithPadding(tokenizer=tokenizer)

    # 이후 토큰을 추가하는 경우 이 부분에 추가해주세요.
    # tokenizer.add_special_tokens()
    # tokenizer.add_tokens()

    if conf.data.tem == 2: #typed entity token에 쓰이는 스페셜 토큰
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>', '<e3>', '</e3>', '<e4>', '</e4>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    # .bin을 가져옵니다.
    load_model_path = conf.path.load_model_path
    checkpoint = torch.load(load_model_path)

    # 모델 구조를 가져옵니다.
    if conf.model.model_class_name == 'TAPT' :
        model = AutoModelForSequenceClassification.from_pretrained(
        conf.path.load_pretrained_model_path, num_labels=30
    )
    else:
        model_class = locate(f'model.model.{conf.model.model_class_name}')
        model = model_class(conf, len(tokenizer))
    ### Refactoring 필요!!


    # 모델 구조 위에 checkpoint를 덮어씌웁니다.
    # 모델 구조와 checkpoint에 저장되어 있는 파라미터 구조가 다른 경우 에러가 발생합니다.
    model.load_state_dict(checkpoint)
    model.parameters
    model.to(device)
    model.eval()

    ## load predict datset
    RE_predict_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.predict_path,conf)
    RE_test_dataset = dataloader.load_dataset(tokenizer, conf.path.test_path,conf)

    # arguments for Trainer
    # predict data를 padding없이 입력하기 위해 batch_size를 1로 입력합니다.
    # batch_size를 키우고 싶은 경우엔 trainer에 collator를 추가해서 실행시켜주세요.
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

        output.to_csv(f"./prediction/dev_submission_{inference_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    if(predict_submit):
        metrics = trainer.evaluate(RE_test_dataset)
        print("Training is complete!")
        print("==================== Test metric score ====================")
        print("eval loss: ", metrics["eval_loss"])
        print("eval auprc: ", metrics["eval_auprc"])
        print("eval micro f1 score: ", metrics["eval_micro f1 score"])

        outputs = trainer.predict(RE_predict_dataset)
        logits = torch.FloatTensor(outputs.predictions)
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
