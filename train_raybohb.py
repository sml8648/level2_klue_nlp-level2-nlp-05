from typing import Optional
import os
import shutil
import json
import utils.util as utils
from datetime import datetime
#import wandb

from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB, ASHAScheduler
from ray.tune import CLIReporter
#from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.logger import DEFAULT_LOGGERS
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import BestRun
from ray.tune.stopper import TrialPlateauStopper
from pathlib import Path


def hyperparameter_tune(trainer: Trainer, training_args: TrainingArguments, experiment_name) -> BestRun:
    resume = False
    def ray_hp_space(trial):
        return {
            #"weight_decay": tune.uniform(0.0, 0.3),
            "num_train_epochs": tune.randint(5,10),
            "learning_rate": tune.loguniform(1e-5, 2e-5),
            "warmup_ratio": tune.uniform(0, 0.5),
            #"attention_probs_dropout_prob": tune.uniform(0, 0.2),
            #"hidden_dropout_prob": tune.uniform(0, 0.2),
            #"per_device_train_batch_size": tune.choice([16]),
        }

    # time_attr = "training_iteration"
    time_attr = "epoch"

    scheduler = HyperBandForBOHB(
        time_attr=time_attr,
        # metric=utils.compute_metrics,
        # number of training_iterations (evaluations) to run for each trial, * 2 to allow for grace period
        # max_t=max_training_iterations * 2,
        max_t=int(training_args.num_train_epochs) * 2,
        reduction_factor=4,
        stop_last_trials=True,
    )

    search = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
        max_concurrent=4
    )

    reporter = CLIReporter(
        parameter_columns={
            #"weight_decay": "w_decay",
            "learning_rate": "lr",
            "warmup_ratio": "wr",
            #"attention_probs_dropout_prob": "att_do",
            #"hidden_dropout_prob": "hi_do",
            #"per_device_train_batch_size": "bs",
            # "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs"
        },
        metric_columns=[
            "eval_auprc", "eval_loss", "eval_micro f1 score",
            "epoch", "training_iteration"
        ])

    def my_objective(metrics):
        return metrics["eval_micro f1 score"]

    best_run = trainer.hyperparameter_search(
        hp_space=ray_hp_space,
        metric="eval_micro f1 score",
        mode="max",
        direction="maximize",
        backend="ray",
        n_trials=10,
        scheduler=scheduler,
        search_alg=search,
        #keep_checkpoints_num=1,
        # checkpoint_score_attr="training_iteration",
        checkpoint_score_attr="epoch",
        stop=TrialPlateauStopper("eval_micro f1 score"),
        progress_reporter=reporter,
        local_dir='./raytune_log',
        name=experiment_name,
        log_to_file=True,
        fail_fast=True,
        resume=resume,
        compute_objective=my_objective
    )

    #save_best_model(best_run, experiment_name, './raytune_log')
    return best_run