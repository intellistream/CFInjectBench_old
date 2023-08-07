# pylint: disable=import-error

import argparse
import json
import random
import torch
import numpy as np

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

    arg_ = parser.parse_args()

    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    # Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    args_dict = dict(
        output_dir=hparam.__dict__.get('output_dir'),
        dataset=hparam.dataset,
        dataset_version=hparam.dataset_version,
        model_name_or_path=hparam.model,
        method=hparam.method,
        freeze_level=hparam.__dict__.get('freeze_level'),
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.__dict__.get('learning_rate'),
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.__dict__.get('train_batch_size'),
        eval_batch_size=hparam.__dict__.get('eval_batch_size'),
        num_train_epochs=hparam.__dict__.get('num_train_epochs'),
        gradient_accumulation_steps=hparam.__dict__.get(
            'gradient_accumulation_steps'),
        n_gpu=hparam.ngpu,
        num_workers=4 * hparam.ngpu,
        use_lr_scheduling=hparam.__dict__.get('use_lr_scheduling'),
        val_check_interval=1.0,
        use_deepspeed=hparam.__dict__.get('use_deepspeed'),
        max_grad_norm=0.5,
        seed=42,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.__dict__.get('checkpoint_path'),
        output_log=hparam.__dict__.get('output_log'),
        alpha=hparam.__dict__.get('alpha'),
        temperature=hparam.__dict__.get('temperature'),
        distil_epoch=hparam.__dict__.get('distil_epoch')
    )

    args = argparse.Namespace(**args_dict)
    Model = load_model('T5')

    set_seed(42)
    train(args, Model)