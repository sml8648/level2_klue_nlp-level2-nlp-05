import pandas as pd
import torch
from utils.util import label_to_num
from tqdm.auto import tqdm

class RE_Dataset(torch.utils.data.Dataset):

  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {k:torch.tensor(v) for k,v in self.pair_dataset[idx].items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def tokenized_dataset(dataset, tokenizer):

  data = []
  for idx, item in tqdm(dataset.iterrows(), desc='tokenizing', total=len(dataset)):

    subj = eval(item['subject_entity'])['word']
    obj = eval(item['object_entity'])['word']

    concat_entity = tokenizer.sep_token.join([subj,obj])
    output = tokenizer(concat_entity,
                      item['sentence'],
                      padding=True,
                      truncation=True,
                      max_length=256,
                      add_special_tokens=True)

    data.append(output)

  return data

def load_train_dataset(tokenizer, train_path):
  train_dataset = pd.read_csv(train_path,index_col=0)

  train_label = label_to_num(train_dataset['label'].values)
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  return RE_train_dataset


def load_dev_dataset(tokenizer, dev_path):

  dev_dataset = pd.read_csv(dev_path,index_col=0)

  dev_label = label_to_num(dev_dataset['label'].values)
  # tokenizing dataset
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
  # make dataset for pytorch.
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  return RE_dev_dataset


def load_test_dataset(tokenizer, test_path):

  test_dataset = pd.read_csv(test_path,index_col=0)

  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label
