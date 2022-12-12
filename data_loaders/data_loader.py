import re
from ast import literal_eval
from typing import Dict, List, Union
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import DataCollatorWithPadding

from data_loaders.preprocessing import add_entity_token
from utils.util import label_to_num


class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.pair_dataset[idx].items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.pair_dataset)


def tokenized_dataset(dataset, tokenizer):
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):

        subj = eval(item["subject_entity"])["word"]
        obj = eval(item["object_entity"])["word"]

        concat_entity = tokenizer.sep_token.join([subj, obj])
        # roberta 모델은 token_type_ids 레이어를 사용하지 않습니다.
        output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
        data.append(output)
    print("========== Tokenized data keys ==========")
    print(data[0].keys())
    return data


def load_dataset(tokenizer, data_path, conf):
    dataset = pd.read_csv(data_path, index_col=0)
    label = label_to_num(dataset["label"].values)
    tokenized_test = dataloader_config[conf.data.dataloader](dataset, tokenizer)
    RE_dataset = RE_Dataset(tokenized_test, label)
    return RE_dataset


def load_predict_dataset(tokenizer, predict_path, conf):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = dataloader_config[conf.data.dataloader](predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset


# tem == 1 : Typed marker, entity marker 사용
def typed_entity_marker_tokenized_dataset(dataset, tokenizer):
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="add_entity_token", total=len(dataset)):
        sent = add_entity_token(item, tem=1)
        output = tokenizer(sent, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False)
        data.append(output)  # [{input_ids, attention_mask, token_type_ids}]

    return data


# tem == 2 : Typed marker, entity marker + emask 사용
def type_entity_marker_emask_tokenized_dataset(dataset, tokenizer):
    """
    1. 스페셜 토큰을 사용하여 typed entity marker 표시
    2. 스페셜 토큰의 위치를 저장
    3. 스페셜 토큰을 특수기호로 치환
    4. 토크나이징
    5. 토큰화한 길이랑 같은 길이면서, (2번에서 기록한)스페셜 토큰의 위치는 1, 나머지는 0인 emask 생성
    """
    data = []

    # typed_entity_marker 사용시 스페셜토큰 추가
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):
        sent = add_entity_token(item, tem=2)
        # 문장을 tokenize 한 후 tokenized_sent 변수에 할당
        tokenized_sent = tokenizer.tokenize(sent)

        # 스페셜토큰 위치 리스트, 스페셜토큰 리스트, 대체토큰 리스트
        e_p_list = []
        s_t_list = ["<e1>", "</e1>", "<e2>", "</e2>", "<e3>", "</e3>", "<e4>", "</e4>"]
        rp_t_list = ["@", "@", "#", "#", "*", "*", "%", "%"]

        # 토큰화된 문장에서의 몇번째 위치인지를 확인
        for s_t in s_t_list:
            e_p_list.append(tokenized_sent.index(s_t))

        # 대체토큰으로 교환
        for s_t, rp_t in zip(s_t_list, rp_t_list):
            sent = re.sub(s_t, rp_t, sent)

        # Add 1 because of the [CLS] token
        for idx in range(len(s_t_list)):
            e_p_list[idx] += 1

        # 문장을 tokenizer setting 에 맞게 tokenize 진행
        tokenized_sentences = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256, add_special_tokens=True)

        # 차원 낮추기
        tokenized_sentences["input_ids"] = tokenized_sentences["input_ids"].squeeze()
        tokenized_sentences["attention_mask"] = tokenized_sentences["attention_mask"].squeeze()
        tokenized_sentences["token_type_ids"] = tokenized_sentences["token_type_ids"].squeeze()

        # special_token 의 위치를 저장하기 위한 배열 생성
        e1_mask = [0] * tokenized_sentences["attention_mask"].shape[0]
        e2_mask = [0] * tokenized_sentences["attention_mask"].shape[0]
        e3_mask = [0] * tokenized_sentences["attention_mask"].shape[0]
        e4_mask = [0] * tokenized_sentences["attention_mask"].shape[0]

        e1_mask[e_p_list[0]] = 1
        e1_mask[e_p_list[1]] = 1
        e2_mask[e_p_list[2]] = 1
        e2_mask[e_p_list[3]] = 1
        e3_mask[e_p_list[4]] = 1
        e3_mask[e_p_list[5]] = 1
        e4_mask[e_p_list[6]] = 1
        e4_mask[e_p_list[7]] = 1

        tokenized_sentences["e1_mask"] = torch.tensor(e1_mask, dtype=torch.long)
        tokenized_sentences["e2_mask"] = torch.tensor(e2_mask, dtype=torch.long)
        tokenized_sentences["e3_mask"] = torch.tensor(e3_mask, dtype=torch.long)
        tokenized_sentences["e4_mask"] = torch.tensor(e4_mask, dtype=torch.long)

        data.append(tokenized_sentences)
    return data


# entity marker만 사용
def entity_marker_tokenized_dataset(dataset, tokenizer):
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="add_entity_token & tokenizing", total=len(dataset)):
        sent = add_entity_token(item, tem=1)
        output = tokenizer(sent, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False)
        sub_token = "@"
        obj_token = "#"
        sub_id = tokenizer.convert_tokens_to_ids(sub_token)
        obj_id = tokenizer.convert_tokens_to_ids(obj_token)
        found = {sub_id: 0, obj_id: 0}
        entity_token = []
        for input_id in output["input_ids"]:
            if input_id in found:
                found[input_id] += 1
                entity_token.append(0)
            elif found[sub_id] == 1 or found[obj_id] == 1:
                entity_token.append(1)
            else:
                entity_token.append(0)

        output["entity_token"] = entity_token
        data.append(output)
    return data


def recent_tokenized_dataset(dataset, tokenizer):
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="add_entity_type_token & tokenizing", total=len(dataset)):
        sent = add_entity_token(item, tem=1)
        se = literal_eval(item["subject_entity"])["type"]
        oe = literal_eval(item["object_entity"])["type"]
        if (se, oe) == ("LOC", "DAT"):
            output.attention_mask[0] = 2
        else:
            pairs = [
                ("ORG", "PER"),
                ("ORG", "ORG"),
                ("ORG", "DAT"),
                ("ORG", "LOC"),
                ("ORG", "POH"),
                ("ORG", "NOH"),
                ("PER", "PER"),
                ("PER", "ORG"),
                ("PER", "DAT"),
                ("PER", "LOC"),
                ("PER", "POH"),
                ("PER", "NOH"),
            ]
            output = tokenizer(sent, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False)
            output.attention_mask[0] = pairs.index((se, oe))
        data.append(output)  # [{input_ids, attention_mask, token_type_ids}]
    return data


dataloader_config = {
    "typed_entity_marker": typed_entity_marker_tokenized_dataset,
    "typed_entity_marker_emask": type_entity_marker_emask_tokenized_dataset,
    "entity_marker": entity_marker_tokenized_dataset,
    "recent": recent_tokenized_dataset,
    "default": tokenized_dataset,
}


class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len = 0
        for i in features:
            if len(i["input_ids"]) > max_len:
                max_len = len(i["input_ids"])

        batch = defaultdict(list)
        for item in features:
            for k in item:
                if "label" not in k:
                    padding_len = max_len - item[k].size(0)
                    if k == "input_ids":
                        item[k] = torch.cat((item[k], torch.tensor([self.tokenizer.pad_token_id] * padding_len)), dim=0)
                    else:
                        item[k] = torch.cat((item[k], torch.tensor([0] * padding_len)), dim=0)
                batch[k].append(item[k])

        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
            batch[k] = batch[k].to(torch.long)
        return batch
