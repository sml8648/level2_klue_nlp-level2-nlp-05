import pandas as pd
import torch
from utils.util import label_to_num
from tqdm.auto import tqdm


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
        # inference를 하는 경우 return_token_type_ids=False 를 반드시 설정해야 입력차원이 학습 때와 같아져 오류가 발생하지 않습니다!!!!!
        output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
        data.append(output)
    return data


def load_dataset(tokenizer, data_path):
    dataset = pd.read_csv(data_path, index_col=0)
    label = label_to_num(dataset["label"].values)
    tokenized_test = tokenized_dataset(dataset, tokenizer)
    RE_dataset = RE_Dataset(tokenized_test, label)
    return RE_dataset


def load_predict_dataset(tokenizer, predict_path):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = tokenized_dataset(predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset
