# -*- coding: utf-8 -*-
""" Main entrance to train ERNIE multitask language model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.distributed import fleet
import numpy as np
import collections
import argparse
import logging
import random
import copy
import json
import os
import time

try:
    import paddlecloud.visual_util as visualdl
except ImportError:
    pass

logging.getLogger().setLevel(logging.INFO)

from senta.common.rule import InstanceName
from senta.models.ernie_multil_task_language_model import ErnieMTLM
from senta.models.ernie_skep_multil_task_language_model import ErnieSkepMTLM
from senta.models.roberta_language_model import RobertaLM
from senta.models.roberta_skep_language_model import RobertaSkepLM
from senta.modules.ernie import ErnieConfig
from senta.common.args import ArgumentGroup, print_arguments
from senta.utils.util_helper import save_infer_data_meta
from senta.data.tokenizer.tokenization_wp import FullTokenizer, GptBpeTokenizer
from senta.data.data_set_reader.ernie_pretrain_dataset_reader import ErniePretrainDataReader
from senta.data.data_set_reader.ernie_skep_pretrain_dataset_reader import ErnieSkepPretrainDataReader
from senta.data.data_set_reader.roberta_pretrain_dataset_reader_en import RobertaPretrainDataReaderEnglish
from senta.data.data_set_reader.roberta_skep_pretrain_dataset_reader_en import RobertaSkepPretrainDataReaderEnglish
import senta.utils.init as init
from senta.utils import log
from senta.training.base_trainer import BaseTrainer

# Argument parsing
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "Model configuration and paths.")
model_g.add_arg("ernie_config_path", str, "./config/ernie_config.json", "Path to the JSON file for ERNIE model config.")
model_g.add_arg("load_checkpoint", str, None, "Path to checkpoint to load (if any).")
model_g.add_arg("load_parameters", str, None, "Path to parameters to load (if any).")
model_g.add_arg("model_type", str, 'ernie_en', "The model architecture to be trained.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("weight_sharing", bool, True, "Whether to share weights between embedding and MLM.")

train_g = ArgumentGroup(parser, "training", "Training options.")
train_g.add_arg("epoch", int, 100, "Number of epochs for training.")
train_g.add_arg("learning_rate", float, 0.0001, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay", "Learning rate scheduler.")
train_g.add_arg("train_batch_size", int, 1024, "Batch size for training.")
train_g.add_arg("eval_step", int, 20, "The step interval to evaluate model performance.")
train_g.add_arg("use_fp16", bool, False, "Whether to use FP16 mixed precision training.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths, and processing options.")
data_g.add_arg("train_filelist", str, "", "Path to training filelist.")
data_g.add_arg("valid_filelist", str, "", "Path to validation filelist.")
data_g.add_arg("test_filelist", str, "", "Path to test filelist.")
data_g.add_arg("vocab_path", str, "", "Path to vocabulary file.")
data_g.add_arg("max_seq_len", int, 512, "Maximum sequence length.")

log_g = ArgumentGroup(parser, "logging", "Logging related.")
log_g.add_arg("log_dir", str, "log", "Directory for logging.")

# Define model classes and configurations
MODEL_CLASSES = {
    "ernie_1.0_ch": (ErnieConfig, ErnieMTLM, FullTokenizer, ErniePretrainDataReader),
    "ernie_2.0_en": (ErnieConfig, ErnieMTLM, FullTokenizer, ErniePretrainDataReader),
    "roberta_en": (ErnieConfig, RobertaLM, GptBpeTokenizer, RobertaPretrainDataReaderEnglish),
}

def main(args):
    """Main function for training and evaluation."""
    log.init_log(os.path.join(args.log_dir, "train"), level=logging.DEBUG)
    config_class, model_class, tokenizer_class, reader_class = MODEL_CLASSES[args.model_type]

    # Load model configuration and tokenizer
    config = config_class(args.ernie_config_path)
    model = model_class(config, args)
    tokenizer = tokenizer_class(vocab_file=args.vocab_path)

    # Initialize dataset readers
    train_reader = reader_class(args, 'train_reader', tokenizer)
    dev_reader = reader_class(args, 'dev_reader', tokenizer, evaluate=True)

    # Configure Paddle Fleet for distributed training
    paddle.distributed.init_parallel_env()

    # Define optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay
    )

    # Train and evaluate
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_reader()):
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if step % args.eval_step == 0:
                logging.info(f"Step {step}, Loss: {loss.numpy()}")

        # Evaluate on dev set
        evaluate(dev_reader, model)


def evaluate(dev_reader, model):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with paddle.no_grad():
        for batch in dev_reader():
            outputs = model(**batch)
            total_loss += outputs["loss"].numpy()
            num_batches += 1

    logging.info(f"Validation Loss: {total_loss / num_batches}")
    model.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    main(args)
