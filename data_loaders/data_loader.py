import pandas as pd
import torch
from utils.util import label_to_num
from tqdm.auto import tqdm
from ast import literal_eval
import re


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

#typed-entity 스페셜 토큰 추가
def add_entity_token(row, tem):   
    '''
    tem == 1 :〈Something〉는 #%PER%조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.
    tem == 2 :〈Something〉는 <e2><e4>PER</e4>조지 해리슨</e2>이 쓰고 <e1><e3>ORG</e3>비틀즈</e1>가 1969년 앨범 《Abbey Road》에 담은 노래다.
    '''
    #entity token list. tem == 1 : 특수기호 토큰, tem == 2 : 스페셜 토큰
    etl=[[],["@", "@", "#", "#", "*", "*", "%", "%"],['<e1>','</e1>','<e2>','</e2>','<e3>','</e3>','<e4>','</e4>']]

    sent = row['sentence']      #sentence
    se = literal_eval(row['subject_entity'])  #subject entity
    oe = literal_eval(row['object_entity'])   #object entity

    new_sent = ''
    if se['start_idx'] < oe['start_idx']: #문장에 subject -> object 순으로 등장
        new_sent = sent[:se['start_idx']] + etl[tem][0]+etl[tem][4]+se['type']+etl[tem][5] + sent[se['start_idx']:se['end_idx'] + 1] + etl[tem][1]  \
                    + sent[se['end_idx'] + 1:oe['start_idx']]+ etl[tem][2]+etl[tem][6]+oe['type']+etl[tem][7]+ sent[oe['start_idx']:oe['end_idx'] + 1] + etl[tem][3] + sent[oe['end_idx'] + 1:]
    else:#문장에 object -> subject 순으로 등장
        new_sent = sent[:oe['start_idx']]+ etl[tem][2]+etl[tem][6]+oe['type']+etl[tem][7]+ sent[oe['start_idx']:oe['end_idx'] + 1] + etl[tem][3] \
                    + sent[oe['end_idx'] + 1:se['start_idx']] + etl[tem][0]+etl[tem][4]+se['type']+etl[tem][5] + sent[se['start_idx']:se['end_idx'] + 1] + etl[tem][1] + sent[se['end_idx'] + 1:]

    return new_sent
    
def tokenized_dataset(dataset, tokenizer,conf):
    data = []
    if conf.data.tem == 1:  # Typed entity marker만 사용
        for _, item in tqdm(dataset.iterrows(), desc="add_entity_type_token & tokenizing", total=len(dataset)):
            sent = add_entity_token(item,conf.data.tem)
            output = tokenizer(sent, padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False)
            data.append(output)     #[{input_ids, attention_mask, token_type_ids}]

    elif conf.data.tem == 2:  # typed entity marker + emask
        '''
        1. 스페셜 토큰을 사용하여 typed entity marker 표시
        2. 스페셜 토큰의 위치를 저장 
        3. 스페셜 토큰을 특수기호로 치환
        4. 토크나이징
        5. 토큰화한 길이랑 같은 길이면서, (2번에서 기록한)스페셜 토큰의 위치는 1, 나머지는 0인 emask 생성
        '''
        sentence_list = []    
        #typed_entity_marker 사용시 스페셜토큰 추가
        for _, item in tqdm(dataset.iterrows(), desc="add_entity_token", total=len(dataset)):
            sentence_list.append(add_entity_token(item,conf.data.tem))

        for sent in tqdm(sentence_list, desc="tokenizing", total=len(sentence_list)):
            # 문장을 tokenize 한 후 tokenized_sent 변수에 할당
            tokenized_sent = tokenizer.tokenize(sent)
            #스페셜토큰 위치 리스트, 스페셜토큰 리스트, 대체토큰 리스트
            e_p_list = []
            s_t_list = ['<e1>','</e1>','<e2>','</e2>','<e3>','</e3>','<e4>','</e4>']
            rp_t_list = ["@", "@", "#", "#", "*", "*", "%", "%"]

            # 토큰화된 문장에서의 몇번째 위치인지를 확인
            for s_t in s_t_list:
                e_p_list.append(tokenized_sent.index(s_t))

            # 대체토큰으로 교환
            for s_t, rp_t in zip(s_t_list,rp_t_list):
                sent = re.sub(s_t,rp_t,sent)

            # Add 1 because of the [CLS] token
            for idx in range(len(s_t_list)):
                e_p_list[idx] += 1
            
            # 문장을 tokenizer setting 에 맞게 tokenize 진행
            tokenized_sentences = tokenizer(
                sent,
                return_tensors="pt",
                padding="max_length",  # collate 사용불가로 인해 padding 사이즈 맞추기
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                return_token_type_ids=False if 'roberta' in conf.model.model_name else True, #roberta는 사용안함, inference시 꼭 False로!
            )
            #차원 낮추기
            tokenized_sentences['input_ids'] = tokenized_sentences['input_ids'].squeeze()
            tokenized_sentences['attention_mask'] = tokenized_sentences['attention_mask'].squeeze()
            #token type ids 사용시
            if 'roberta' not in conf.model.model_name:
                tokenized_sentences['token_type_ids'] = tokenized_sentences['token_type_ids'].squeeze()
            
            # special_token 의 위치를 저장하기 위한 배열 생성
            e1_mask = [0] * tokenized_sentences['attention_mask'].shape[0]
            e2_mask = [0] * tokenized_sentences['attention_mask'].shape[0]
            e3_mask = [0] * tokenized_sentences['attention_mask'].shape[0]
            e4_mask = [0] * tokenized_sentences['attention_mask'].shape[0]
            
            e1_mask[e_p_list[0]] = 1
            e1_mask[e_p_list[1]] = 1
            e2_mask[e_p_list[2]] = 1
            e2_mask[e_p_list[3]] = 1
            e3_mask[e_p_list[4]] = 1
            e3_mask[e_p_list[5]] = 1
            e4_mask[e_p_list[6]] = 1
            e4_mask[e_p_list[7]] = 1

            tokenized_sentences['e1_mask'] = torch.tensor(e1_mask, dtype=torch.long)
            tokenized_sentences['e2_mask'] = torch.tensor(e2_mask, dtype=torch.long)
            tokenized_sentences['e3_mask'] = torch.tensor(e3_mask, dtype=torch.long)
            tokenized_sentences['e4_mask'] = torch.tensor(e4_mask, dtype=torch.long)
            
            data.append(tokenized_sentences)    #[{input_ids, attention_mask, token_type_ids, e1mask, e2mask, e3mask, e4mask}]
    else:
        for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):

            subj = eval(item["subject_entity"])["word"]
            obj = eval(item["object_entity"])["word"]

            concat_entity = tokenizer.sep_token.join([subj, obj])
            # roberta 모델은 token_type_ids 레이어를 사용하지 않습니다.
            # inference를 하는 경우 return_token_type_ids=False 를 반드시 설정해야 입력차원이 학습 때와 같아져 오류가 발생하지 않습니다!!!!!
            output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
            data.append(output)
    print("========== Tokenized data keys ==========")
    print(data[0].keys())
    return data


def load_dataset(tokenizer, data_path, conf):
    dataset = pd.read_csv(data_path, index_col=0)
    label = label_to_num(dataset["label"].values)
    tokenized_test = tokenized_dataset(dataset, tokenizer, conf)
    RE_dataset = RE_Dataset(tokenized_test, label)
    return RE_dataset


def load_predict_dataset(tokenizer, predict_path, conf):
    predict_dataset = pd.read_csv(predict_path, index_col=0)
    predict_label = None
    tokenized_predict = tokenized_dataset(predict_dataset, tokenizer, conf)
    RE_predict_dataset = RE_Dataset(tokenized_predict, predict_label)
    return RE_predict_dataset