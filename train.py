import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
# https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html
import data_loaders.data_loader as dataloader
import utils.util as utils
import transformers
from torch.optim.lr_scheduler import OneCycleLR

import model.model as model_arch
from transformers import DataCollatorWithPadding
# https://huggingface.co/course/chapter3/4

import mlflow
import mlflow.sklearn
from azureml.core import Workspace
import argparse

from omegaconf import OmegaConf




def start_mlflow(experiment_name):
  #Enter details of your AzureML workspace
  subscription_id = "0275dc6c-996d-42d1-8263-8f7b4e81f271"
  resource_group = "basburger"
  workspace_name = "basburger"
  ws = Workspace.get(name=workspace_name,
                    subscription_id=subscription_id,
                    resource_group=resource_group)

  mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

  # https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-log-view-metrics?tabs=interactive
  mlflow.set_experiment(experiment_name)
  # Start the run
  mlflow_run = mlflow.start_run()

def train(args, conf):
  #huggingface-cli login  #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

  model_name = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
  if args.tem: #typed entity token에 쓰이는 스페셜 토큰
    special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>', '<e3>', '</e3>', '<e4>', '</e4>']}
    tokenizer.add_special_tokens(special_tokens_dict)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  
  new_token_count = 0
  # new_token_count += tokenizer.add_special_tokens()
  # new_token_count += tokenizer.add_tokens()
  new_vocab_size = tokenizer.vocab_size + new_token_count

  experiment_name = model_name+'_bs'+str(conf.train.batch_size)+'_ep'+str(conf.train.max_epoch)+'_lr'\
    +str(conf.train.learning_rate)
  # start_mlflow(experiment_name)

  # load dataset
  RE_train_dataset = dataloader.load_train_dataset(tokenizer, conf.path.train_path, args)
  RE_dev_dataset = dataloader.load_dev_dataset(tokenizer, conf.path.dev_path, args)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load model
  #model = model_arch.Model(args, conf, new_vocab_size)
  
  #RBERT
  model_config = AutoConfig.from_pretrained(model_name)
  model = model_arch.CustomRBERT(model_config, model_name)
  model.model.resize_token_embeddings(len(tokenizer))

  model.parameters
  model.to(device)
  optimizer = transformers.AdamW(model.parameters(), lr=conf.train.learning_rate)
  scheduler = OneCycleLR(optimizer, max_lr=conf.train.learning_rate, steps_per_epoch=len(RE_train_dataset)//conf.train.batch_size+1,
                         pct_start=0.5, epochs=conf.train.max_epoch, anneal_strategy='linear', div_factor=1e100, final_div_factor=1)

  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    hub_model_id="jstep750/basburger",
    output_dir='./output',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=914,                 # model saving step.
    num_train_epochs=conf.train.max_epoch,              # total number of training epochs
    learning_rate=conf.train.learning_rate,               # learning_rate
    per_device_train_batch_size=conf.train.batch_size,  # batch size per device during training
    per_device_eval_batch_size=conf.train.batch_size,   # batch size for evaluation
    # weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 914,            # evaluation step.
    load_best_model_at_end=True,
    push_to_hub=True,
    metric_for_best_model='micro f1 score'
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=utils.compute_metrics,         # define metrics function
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=conf.utils.patience)]
  )

  # train model
  trainer.train()
  trainer.save_model('./best_model')
  mlflow.end_run()
  trainer.push_to_hub()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")
  parser.add_argument('--shuffle', default=True)
  # parser.add_argument('--optimizer', default='AdamW')

  parser.add_argument('--preprocessing', default=False)
  parser.add_argument('--precision', default=32, type=int)
  parser.add_argument('--dropout', default=0.1, type=float)
  parser.add_argument('--tem', default=True, type=bool)
  args = parser.parse_args()
  
  conf = OmegaConf.load(f"./config/{args.config}.yaml")
  # check hyperparameter arguments
  print(args)
  train(args, conf)