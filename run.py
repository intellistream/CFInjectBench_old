# pylint: disable=import-error

import os
import shutil
import argparse
import json
import random
import torch
import numpy as np

from online_evaluation import evaluate
from online_train import train
from models import load_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--model', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    # Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    # Init configs that are not given
    if 'split_num' not in hparam:
        hparam.split_num = 1
    if 'split' not in hparam:
        hparam.split = 0
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'output_log' not in hparam:
        hparam.output_log = None
    if 'stream_mini_batch_size' not in hparam:
        hparam.stream_mini_batch_size = 0
    if 'eval_batch_size' not in hparam:
        hparam.eval_batch_size = 1

    # Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir,  # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version=hparam.dataset_version,
        split_num=hparam.split_num,
        split=hparam.split,
        model_name_or_path=hparam.model,
        method=hparam.method,
        freeze_level=hparam.freeze_level,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        warmup_steps=0,
        stream_mini_batch_size=hparam.stream_mini_batch_size,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=4 * hparam.ngpu,
        use_lr_scheduling=hparam.use_lr_scheduling,
        val_check_interval=1.0,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        use_deepspeed=hparam.use_deepspeed,
        opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        max_grad_norm=hparam.grad_norm,
        seed=42,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)

    # Getting the Model type & Method
    if 't5' in args.model_name_or_path:
        model_type = 'T5'
    # elif 'gpt2' in args.model_name_or_path:
    #     model_type = 'GPT2'
    else:
        raise Exception(
            'Select the correct model. Supporting "t5" and "gpt2" only.')

    Model = load_model(model_type)

    if args.check_validation_only:
        evaluate(args, arg_.model, Model)
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

        set_seed(40)
        train(args, Model)
