import pandas as pd
import torch
from utils.util import label_to_num
from ast import literal_eval 

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
  

def preprocessing_dataset(dataset, args):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
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


def load_data(dataset_dir, args):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset,args)
  return dataset

def tokenized_dataset(dataset, tokenizer, args):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  if args.tem is False:
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
  else: # typed entity marker 적용시
    # special token 의 위치를 저장하기 위한 빈 list
    e_p_list = []
    for sent in dataset.sentence:
      # 문장을 tokenize 한 후 tokenized_sent 변수에 할당
      tokenized_sent = tokenizer.tokenize(sent)

      # 토큰화된 문장에서의 몇번째 위치인지를 확인
      e11_p = tokenized_sent.index('<e1>')  # the start position of entity1
      e12_p = tokenized_sent.index('</e1>')  # the end position of entity1
      e21_p = tokenized_sent.index('<e2>')  # the start position of entity2
      e22_p = tokenized_sent.index('</e2>')  # the end position of entity2
      e31_p = tokenized_sent.index('<e3>')  # the start position of entity3
      e32_p = tokenized_sent.index('</e3>')  # the end position of entity3
      e41_p = tokenized_sent.index('<e4>')  # the start position of entity4
      e42_p = tokenized_sent.index('</e4>')  # the end position of entity4

      # Replace the token
      tokenized_sent[e11_p] = "@"
      tokenized_sent[e12_p] = "@"
      tokenized_sent[e21_p] = "#"
      tokenized_sent[e22_p] = "#"
      tokenized_sent[e31_p] = "*"
      tokenized_sent[e32_p] = "*"
      tokenized_sent[e41_p] = "∧"
      tokenized_sent[e42_p] = "∧"

      # Add 1 because of the [CLS] token
      e11_p += 1
      e12_p += 1
      e21_p += 1
      e22_p += 1
      e31_p += 1
      e32_p += 1
      e41_p += 1
      e42_p += 1

      # 토큰화된 문장에서 special_token 의 위치를 저장
      e_p_list.append([e11_p, e12_p, e21_p, e22_p, e31_p, e32_p, e41_p, e42_p])

    # 문장을 tokenizer setting 에 맞게 tokenize 진행
    tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      # concat_entity,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=True #roberta는 사용안함,
    )

    # special_token 의 위치를 저장하기 위한 배열 생성
    e1_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                for _ in range(tokenized_sentences['attention_mask'].shape[0])]
    e2_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                for _ in range(tokenized_sentences['attention_mask'].shape[0])]
    e3_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                for _ in range(tokenized_sentences['attention_mask'].shape[0])]
    e4_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                for _ in range(tokenized_sentences['attention_mask'].shape[0])]

    # special_token 의 위치인 곳에 1을 넣어주고 나머지는 0으로 유지
    for i, e_p in enumerate(e_p_list):
      # '#', '*', '@', '∧' 토큰 output vector 만을 사용하는 방법
      e1_mask[i][e_p[0]] = 1
      e1_mask[i][e_p[1]] = 1
      e2_mask[i][e_p[2]] = 1
      e2_mask[i][e_p[3]] = 1
      e3_mask[i][e_p[4]] = 1
      e3_mask[i][e_p[5]] = 1
      e4_mask[i][e_p[6]] = 1
      e4_mask[i][e_p[7]] = 1

    # 최종 return 되는 dictionary 형태의 데이터에 special token mask 배열을 tensor 로 변경해 추가
    tokenized_sentences['e1_mask'] = torch.tensor(e1_mask, dtype=torch.long)
    tokenized_sentences['e2_mask'] = torch.tensor(e2_mask, dtype=torch.long)
    tokenized_sentences['e3_mask'] = torch.tensor(e3_mask, dtype=torch.long)
    tokenized_sentences['e4_mask'] = torch.tensor(e4_mask, dtype=torch.long)
  
  return tokenized_sentences


def load_train_dataset(tokenizer, train_path, args):
  train_dataset = load_data(train_path, args)
  train_label = label_to_num(train_dataset['label'].values)
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer,args)
  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  return RE_train_dataset


def load_dev_dataset(tokenizer, dev_path,args):
  dev_dataset = load_data(dev_path, args) # validation용 데이터는 따로 만드셔야 합니다.
  dev_label = label_to_num(dev_dataset['label'].values)
  # tokenizing dataset
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer,args)
  # make dataset for pytorch.
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  return RE_dev_dataset


def load_test_dataset(tokenizer, test_path,args):
  test_dataset = load_data(test_path,args)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer,args)
  return test_dataset['id'], tokenized_test, test_label

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