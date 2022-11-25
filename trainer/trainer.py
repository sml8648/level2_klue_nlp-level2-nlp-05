import numpy as np
import torch
from torchvision.utils import make_grid
import transformers
from transformers import Trainer
from torch.optim.lr_scheduler import OneCycleLR

class CustomTrainer(Trainer):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf

    def create_optimizer(self):
        self.optimizer = transformers.AdamW(self.model.parameters(), lr=self.conf.train.learning_rate)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.conf.train.learning_rate,
            steps_per_epoch=len(self.train_dataset) // self.conf.train.batch_size + 1,
            pct_start=0.5,
            epochs=self.conf.train.max_epoch,
            anneal_strategy="linear",
            div_factor=1e100,
            final_div_factor=1,
        )
        return self.lr_scheduler