import argparse
from tqdm import tqdm
import data_loaders.data_loader as data_loader
import utils.util as util
import model.model as custom

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[1]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  MODEL_NAME = args.model_name # model name.
  model = custom.CustomModel(MODEL_NAME, 30)
  #model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=30)
  checkpoint = torch.load('./output/checkpoint-500/pytorch_model.bin')
  #print(checkpoint.keys())
  model.load_state_dict(checkpoint)
  #model = model.load_from_checkpoint()
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset = "test_data"
  test_id, test_dataset, test_label = data_loader.load_test_dataset(tokenizer, test_dataset)
  Re_test_dataset = data_loader.RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = util.num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', default='klue/roberta-base', type=str)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--max_epoch', default=5, type=int)
  parser.add_argument('--shuffle', default=True)

  parser.add_argument('--learning_rate', default=5e-5, type=float)
  parser.add_argument('--train_data', default='train')
  parser.add_argument('--dev_data', default='dev')
  parser.add_argument('--test_data', default='dev')
  parser.add_argument('--predict_data', default='test')
  parser.add_argument('--optimizer', default='AdamW')
  parser.add_argument('--loss_function', default='L1Loss')

  parser.add_argument('--preprocessing', default=False)
  parser.add_argument('--precision', default=32, type=int)
  parser.add_argument('--dropout', default=0.1, type=float)
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./output")
  args = parser.parse_args()
  print(args)
  main(args)
  
