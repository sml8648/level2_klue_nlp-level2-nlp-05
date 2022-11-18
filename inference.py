from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

import data_loaders.data_loader as dataloader
import model.model as model_arch
import utils.util as utils

import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf


def inference(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = conf.model.model_name

    new_token_count = 0
    # new_token_count += tokenizer.add_special_tokens()
    # new_token_count += tokenizer.add_tokens()
    new_vocab_size = tokenizer.vocab_size + new_token_count

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    ## # loading the model you previously trained
    model_load_path = f"./output/checkpoint-{conf.model.load_checkout}/pytorch_model.bin"  # load model dir.
    checkpoint = torch.load(model_load_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_path, num_labels=30)
    model.resize_token_embeddings(new_vocab_size)
    model.load_state_dict(checkpoint, strict=False)
    model.parameters
    model.to(device)

    ## load test datset
    predict_dataset_dir = conf.path.predict_path
    Re_predict_dataset, predict_id = dataloader.load_predict_dataset(predict_dataset_dir, tokenizer)

    # arguments for Trainer
    test_args = TrainingArguments(output_dir=model_load_path, do_train=False, do_predict=True, per_device_eval_batch_size=conf.train.batch_size, dataloader_drop_last=False)

    # init trainer
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics)

    outputs = trainer.predict(Re_predict_dataset)

    logits = outputs[1]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    pred_answer = result.tolist()
    pred_answer = utils.num_to_label(pred_answer)
    output_prob = prob.tolist()

    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv("./prediction/submission.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args = parser.parse_args()

    conf = OmegaConf.load(f"./config/{args.config}.yaml")
    # check hyperparameter arguments
    print(args)
    inference(conf)
