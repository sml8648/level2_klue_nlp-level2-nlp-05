import data_loaders.data_loader as dataloader

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
import torch
from transformers import Trainer, TrainingArguments

def tapt_pretrain(conf):
    model_name = conf.model.model_name

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # label 없이 가져오기 위해서 load_predict_dataset 사용
    ### Refactoring 필요! ###
    RE_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.pretrain_path, conf)

    # Pretrained model for MaskedLM training
    model_config = AutoConfig.from_pretrained(model_name)  # 모델 가중치 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    # token 15% 확률 masking 진행
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # TAPT task이기 때문에 evaluation_strategy X
    # cuda out-of-memory 발생하여 fp16 = True 로 변경
    training_args = TrainingArguments(
        output_dir="./klue-roberta-pretrained",
        learning_rate=3e-05,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_steps=4000,
        save_total_limit=3,
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=4000,
        fp16=True, # 16비트로 변환
        fp16_opt_level="O1",
        resume_from_checkpoint=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("./klue-roberta-pretrained")  # pretrained_model save
