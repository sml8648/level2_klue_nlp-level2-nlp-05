import pandas as pd
import torch
from utils.util import label_to_num
from ast import literal_eval 
from tqdm.auto import tqdm

class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.pair_dataset[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def tokenized_dataset(dataset, tokenizer):
  if args.tem ==True:
    sentence_list = []    #typed_entity_marker에서 사용
    subject_entity = []
    object_entity = []
    for _, row in dataset.iterrows():
      if args.tem:
        sentence_list.append(add_entity_token(row))
      subject_entity.append(eval(row['subject_entity'])['word'])
      object_entity.append(eval(row['object_entity'])['word'])  

    if args.tem is False: #typed_entity_marker 사용안한다면 원래대로
      sentence_list = dataset['sentence']   

    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentence_list, 'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], })

    return out_dataset
  else:
    data = []
    for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):

        subj = eval(item["subject_entity"])["word"]
        obj = eval(item["object_entity"])["word"]

        concat_entity = tokenizer.sep_token.join([subj, obj])
        # roberta 모델은 token_type_ids 레이어를 사용하지 않습니다.
        # inference를 하는 경우 return_token_type_ids=False 를 반드시 설정해야 입력차원이 학습 때와 같아져 오류가 발생하지 않습니다!!!!!
        output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False)

        data.append(output)

    return data


def load_train_dataset(tokenizer, train_path):
    train_dataset = pd.read_csv(train_path, index_col=0)
    train_label = label_to_num(train_dataset["label"].values)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    return RE_train_dataset


def load_dev_dataset(tokenizer, dev_path):
    dev_dataset = pd.read_csv(dev_path, index_col=0)
    dev_label = label_to_num(dev_dataset["label"].values)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    return RE_dev_dataset


def load_test_dataset(tokenizer, test_path):
    test_dataset = pd.read_csv(test_path, index_col=0)
    test_label = label_to_num(test_dataset["label"].values)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    RE_test_dataset = RE_Dataset(tokenized_test, test_label)
    return RE_test_dataset


def load_predict_dataset(tokenizer, predict_path):
    predict_dataset = pd.read_csv(predict_path)
    predict_id = predict_dataset["id"]
    predict_dataset = predict_dataset.drop("id", axis=1)
    predict_label = list(map(int, predict_dataset["label"].values))
    # tokenizing dataset
    tokenized_predict = tokenized_dataset(predict_dataset, tokenizer)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset, predict_id

#typed-entity 토큰 추가
def add_entity_token(row):
  '''
  before
  〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.,
  "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}",
  "{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}"
  
  after
  〈Something〉는 <e2> <e3> PER </e3> 조지 해리슨 </e2> 이 쓰고 <e1> <e3> ORG </e3> 비틀즈 </e1> 가 1969년 앨범 《Abbey Road》에 담은 노래다
  '''
  sent = row['sentence']      #sentence
  se = literal_eval(row['subject_entity'])  #subject entity
  oe = literal_eval(row['object_entity'])   #object entity

  # 새로운 new_sent 변수에 special_token 추가해서 저장
  # 이때, typed_entity_marker 를 적용할 수 있도록 <e1>, </e1>, <e2>, </e2>, <e3>, </e3>, <e4>, </e4> token 추가하고
  # subject_entity 와 object_entity 의 type 을 new_sent 에 추가해줌
  new_sent = ''
  if se['start_idx'] < oe['start_idx']: #문장에 subject -> object 순으로 등장
      new_sent = sent[:se['start_idx']] + '<e1> <e3> '+se['type']+' </e3> ' + sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> '  \
                  + sent[se['end_idx'] + 1:oe['start_idx']]+ '<e2> <e4> '+oe['type']+' </e4> '+ sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> ' + sent[oe['end_idx'] + 1:]
  else:#문장에 object -> subject 순으로 등장
      new_sent = sent[:oe['start_idx']]+ '<e2> <e4> '+oe['type']+' </e4> '+ sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> ' \
                  + sent[oe['end_idx'] + 1:se['start_idx']] + '<e1> <e3> '+se['type']+' </e3> ' + sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> ' + sent[se['end_idx'] + 1:]

  return new_sent