# pylint: disable=import-error

import os
import csv
import shutil
import time
import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import T5Tokenizer
from collections import deque

from dataset import CKLDataset
from online_evaluation import evaluate
from utils import load_dataset


def train(args, Model):
    train_stream_df = load_dataset('train', args)
    test_stream_df = load_dataset('test', args)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    model = Model(args)

    start_time = time.time()
    collector = []

    trainer = pl.Trainer(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu',
        max_epochs=args.num_train_epochs,
        precision=16 if args.use_deepspeed else 32,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        callbacks=[ModelCheckpoint(
            dirpath=args.output_dir, save_top_k=-1)],
        strategy='ddp'
    )

    last_entry = None 

    bwt = []
    acc = []
    eval_time = []

    periods = deque(maxlen=2)

    output_folder = ("/".join((args.output_log.split('/'))[:-1]))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    for idx, row in train_stream_df.iterrows():
        if last_entry and last_entry != row['date'] or idx == len(train_stream_df) - 1:
            print('Training -', last_entry)

            model.set_dataset(CKLDataset(
                collector, 'train', tokenizer, args))
            trainer.fit(model)
            trainer.fit_loop.max_epochs += args.num_train_epochs

            collector = []
            periods.append(last_entry)

            if trainer.global_rank == 0:
                metrics, e_time = evaluate(
                    args, model, test_stream_df[test_stream_df['date'].isin(periods)], tokenizer)

                if len(periods) == 2:
                    bwt.append(metrics[0] - acc[-1])
                    print('BWT:', bwt[-1])

                acc.append(metrics[-1])
                eval_time.append(e_time)
                print('ACC:', acc[-1])
                print('TIME:', eval_time[-1])

            trainer.strategy.barrier()

        collector.append(row.to_dict())
        last_entry = row['date']

    if trainer.global_rank == 0:
        total_time = time.time() - start_time
        print('Total time:', total_time)

        with open(f'{args.output_log}all.csv', 'w', newline='', encoding='utf-8') as writefile:
            writer = csv.writer(writefile)
            writer.writerows([acc,[0] + bwt, eval_time])

        with open(f'{args.output_log}results.csv', 'w', newline='', encoding='utf-8') as writefile:
            writer = csv.writer(writefile)
            writer.writerow(['ACC', 'BWT', 'TIME', 'TRAIN_TIME'])
            writer.writerow([sum(acc)/len(acc), sum(bwt)/len(bwt), total_time, total_time-sum(eval_time)])

    trainer.strategy.barrier()


