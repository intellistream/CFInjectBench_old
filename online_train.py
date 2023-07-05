# pylint: disable=import-error

import time
import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import T5Tokenizer

from dataset import CKLDataset
from utils import load_dataset


def train(args, Model):
    if args.mode in ['pretrain', 'finetune']:
        stream_dataset, _ = load_dataset('train', args)

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = Model(args)

    start_time = time.time()

    collector = []

    trainer = pl.Trainer(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins='deepspeed_stage_2' if args.use_deepspeed else [],
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.use_deepspeed else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=True,
        val_check_interval=args.val_check_interval,
        callbacks=[ModelCheckpoint(
            dirpath=args.output_dir, save_top_k=-1, period=1)],
        accelerator=args.accelerator,
    )

    for idx, row in stream_dataset.iterrows():
        collector.append(row.to_dict())

        if len(collector) >= args.stream_mini_batch_size or idx == len(stream_dataset) - 1:

            model.set_dataset(CKLDataset(collector, 'train', tokenizer, args))
            trainer.fit(model)

            trainer.max_epochs += (args.num_train_epochs - 1)

            collector = []

    print('Total time:', time.time() - start_time)
