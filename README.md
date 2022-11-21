# 🏅 KLUE Competition - Relation Extraciton


## 📋 Table of contents

* [📝 Competition Description](#competition)
* [💾 Dataset Description](#dataset)
* [🗄 Folder Structure](#folder)
* [⚙️ Set up](#setup)
* [💻 How to Run](#torun)
<br><br/>

---

<br>

## 📝 Competition Description <a name='competition'></a>

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 

이번 대회에서는 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 


<br><br>

## 💾 Dataset Description <a name='dataset'></a>
 
| Dataset            | train                    | test |
| ------------------ | ----------------------- |--------------- |
| **문장 수**        | 32470      |     7765   |
| **비율**        | 80      |     20 |

<br/>

### Columns
* **id** (문자열) : 문장 고유 ID 

* **sentence** (문자열) : 주어진 문장

* **subject_entity** (딕셔너리) : 주체 entity

* **object_entity** (딕셔너리) : 객체 entity

* **label** : (문자열) 30가지 label에 해당하는 주체와 객체간 관계

* **source** : (문자열) 문장의 출처

    * **wikipedia** (위키피디아)

    * **wikitree** (위키트리)

    * **policy_briefing** (정책 보도 자료?)

<br><br>

## 🗄 Folder Structure <a name='folder'></a>
```
├──📁base 
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├──📁config
│   └── base_config.yaml → custom_config.yaml 만들기 가능
├──📁data_loaders
│   └── data_loader.py → 데이터셋을 로드합니다. 
│
├──📁dataset
│   ├──📁dev
│   │   └── dev.csv → dev(valid) 데이터
│   ├──📁predict
│   │   ├── predict.csv → 예측해야하는 데이터
│   │   └── sample_submission.csv → 샘플 데이터
│   ├──📁test
│   │   └── test.csv → 모델 학습 후 마지막 평가에서 사용하는 데이터
│   └──📁train
│       └── train.csv → 학습 데이터
│
├── dict_label_to_num.pkl
├── dict_num_to_label.pkl
├── inference.py → inference 코드
│
├── main.py → train.py와 inference.py 실행 코드
주의❗️) omegaconfig 파일 이름을 main.py에서 입력해야해요 !!!
ex) train하는 경우 → python main.py -mt
    inference하는 경우 → python main.py -mi
│
├──📁model
│   ├── loss.py
│   ├── metric.py 
│   └── model.py
│
├──📁prediction
│   ├── sample_submission.csv
│   ├── submission.csv
│   └── submission_18-14-46.csv → inference하는 경우, '날짜-시간-분.csv'가 뒤에 붙음
│
├──📁step_saved_model → save_steps 조건에서 모델이 저장되는 경로.
│   └──📁klue-roberta-small → 사용한 모델
│       └──📁18-14-42       → 실행한 날짜-시간-분
│           └── checkpoint-500 → 저장된 체크포인트-스탭
│               ├── optimizer.pt
│               ├── pytorch_model.bin → 이 파일을 inference할 때 불러오기!
│               ├── rng_state.pth
│               ├── scheduler.pt
│               ├── trainer_state.json
│               └── training_args.bin
│
├── train.py → train 하는 함수
│
├──📁trainer
│   └── trainer.py
│
└──📁utils
    └── util.py


```

<br><br>

## ⚙️ Set up <a name='setup'></a>

### 1. Requirements

```bash
$ pip install -r requirements.txt
```

### 2. Prepare Dataset - train data split
train : dev : test = 8 : 1 : 1

<br><br>

## 💻 How to Run <a name='torun'></a>

### How to train

```bash
$ python main.py  -mt
```

### How to Inference

```bash
$ python inference.py  -mi
```
