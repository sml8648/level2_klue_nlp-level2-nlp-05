import pandas as pd
import torch
from utils.util import label_to_num

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
  )
  assert '[SEP]' in tokenizer.special_tokens_map.values(), "This tokenizer does not use '[SEP]' token."
  return tokenized_sentences


def load_train_dataset(tokenizer, train_data):
  train_dataset = load_data(f"../dataset/train/{train_data}.csv")
  train_label = label_to_num(train_dataset['label'].values)
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  return RE_train_dataset


def load_dev_dataset(tokenizer, dev_data):
  dev_dataset = load_data(f"../dataset/dev/{dev_data}.csv") # validation용 데이터는 따로 만드셔야 합니다.
  dev_label = label_to_num(dev_dataset['label'].values)
  # tokenizing dataset
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
  # make dataset for pytorch.
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  return RE_dev_dataset


def load_test_dataset(tokenizer, test_data):
  test_dataset = load_data(f"../dataset/test/{test_data}.csv")
  test_label = label_to_num(test_dataset['label'].values)
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  # make dataset for pytorch.
  RE_test_dataset = RE_Dataset(tokenized_test, dev_label)
  return RE_test_dataset