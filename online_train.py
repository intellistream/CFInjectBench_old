# pylint: disable=import-error

import os
import csv
import shutil
import time
import pytorch_lightning as pl
from transformers import T5Tokenizer

from dataset import CKLDataset as Dataset
from kilm_dataset import CKLDataset as KILM_Dataset
from online_evaluation import evaluate
from utils import load_dataset
import numpy as np


def find_next_entry(start_idx, train_stream_df):
    next_entry = None
    for idx, row in train_stream_df.iterrows():
        if idx >= start_idx:
            if next_entry and next_entry != row['date'] or idx == len(train_stream_df) - 1:
                return next_entry
            next_entry = row['date']
    return None

def train(args, Model):
    output_folder = ("/".join((args.output_log.split('/'))[:-1]))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    train_stream_df = load_dataset('train', args)
    test_stream_df = load_dataset('test', args)

    model = Model(args)
    if 't5' in args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = model.tokenizer
    start_time = time.time()
    collector = []

    trainer = pl.Trainer(
        logger=False,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu',
        enable_progress_bar=False,
        max_epochs=args.num_train_epochs,
        precision=16 if args.use_deepspeed else 32,
        devices=args.n_gpu,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        callbacks=[],
        enable_checkpointing=False,
        # callbacks=[CustomModelCheckpoint(dirpath=args.output_dir)],
        strategy='ddp'
    )

    last_entry = None 

    bwt = []
    fwt = []
    acc = []
    pre_metric = []
    train_time = []

    periods = []
    # periods = deque(maxlen=3)
    first_time = True

    writefile = open(f'{args.output_log}results.csv', 'a', newline='', encoding='utf-8')
    acc_writefile = open(f'{args.output_log}acc.csv', 'a', newline='', encoding='utf-8')
    writer = csv.writer(writefile)
    acc_writer = csv.writer(acc_writefile)
    writer.writerow(["Date", "EM", "BWT", "FWT", "Time"])
    acc_writer.writerow(["Date", "EM"])
    writefile.flush()
    acc_writefile.flush()
    flag = True

    if args.method == 'kilm':
        CKLDataset = KILM_Dataset
    else:
        CKLDataset = Dataset

    for idx, row in train_stream_df.iterrows():
        if last_entry and last_entry != row['date'] or idx == len(train_stream_df) - 1:
            repeat_num = args.repeat_num
            if args.model_name_or_path != 'initial':
                if args.model_name_or_path != 'initial':
                    model.set_dataset(CKLDataset(collector, 'train', tokenizer, args))
            if trainer.global_rank == 0:
                print('=' * 50)
                print('=' * 50)
                print('Training -', last_entry)
                print(f"Repeating number: {repeat_num}")
                print(f"Coreset method: {args.coreset}")
                print(f"Coreset ratio: {args.coreset_ratio}")
                start_train = time.time()
            if args.method != 'initial':
                trainer.fit(model)
                trainer.fit_loop.max_epochs += args.num_train_epochs
            if trainer.global_rank == 0:
                train_time.append(time.time() - start_train)
                print(f'TRAIN TIME:{train_time[-1]}')

            collector = []

            if first_time:
                periods.append(last_entry)
            next_entry = find_next_entry(idx, train_stream_df)
            # for the end of the train stream, periods = ['date(t-1)', 'date(t)', None]
            periods.append(next_entry)

            if trainer.global_rank == 0:
                if args.method != 'initial':
                    print(f"For periods: {periods}")
                    metrics, e_time = evaluate(args, model, test_stream_df[test_stream_df['date'].isin(periods)],
                                       tokenizer, trainer.global_rank)
                    metrics = np.array(metrics)
                    if len(periods) > 2:
                        # metric, knowledge:
                        # 1-1 1-2
                        # 2-1 2-2 2-3
                        # 3-1 3-2 3-3 3-4
                        # 4-1 4-2 4-3 4-4 None
                        diff = metrics[:len(pre_metric)][-2:] - pre_metric[-2:]
                        bwt_res = diff[0]
                        fwt_res = diff[1]
                        # if len(pre_metric) == 2:
                        #     bwt_res = metrics[0] - pre_metric[0]
                        #     fwt_res = metrics[1] - pre_metric[1]
                        # else:
                        #     bwt_res = metrics[0] - pre_metric[1]
                        #     fwt_res = metrics[1] - pre_metric[2]

                        bwt.append(bwt_res)
                        fwt.append(fwt_res)

                        print('BWT:', bwt_res)
                        print('FWT:', fwt_res)

                    # dtw.append(dtw_res)
                    pre_metric = metrics
                    # if len(metrics) == 2:
                    #     acc_idx = 0
                    # else:
                    #     acc_idx = 1
                    acc.append(metrics[-2])

                    print('ACC:', acc[-1])
                    # print('TIME:', eval_time[-1])

                    writer = csv.writer(writefile)
                    acc_writer = csv.writer(acc_writefile)
                    # writer.writerow(["Date", "EM", "BWT", "FWT", "Time"])

                    if first_time:
                        writer.writerow([periods[-2], acc[-1], None, None, train_time[-1]])
                        first_time = False
                        writefile.flush()

                        acc_writer.writerow(metrics[:-1])
                        acc_writefile.flush()
                    else:
                        writer.writerow([periods[-2], acc[-1], bwt[-1], fwt[-1], train_time[-1]])
                        writefile.flush()

                        acc_writer.writerow(metrics[:-1])
                        acc_writefile.flush()
                else:
                    if idx == len(train_stream_df) - 1:
                        print(f"For periods: {periods}")
                        metrics, e_time = evaluate(args, model, test_stream_df[test_stream_df['date'].isin(periods)],
                                           tokenizer, trainer.global_rank)
                        metrics = np.array(metrics)
                        acc_writer.writerow(metrics[:-1])
                        acc_writefile.flush()

            trainer.strategy.barrier()

        # #============== control the start date =============
        # if row['date'] == '2019-8':
        #     flag = False
        # if flag:
        #     continue
        collector.append(row.to_dict())
        last_entry = row['date']

    trainer.strategy.barrier()

    if trainer.global_rank == 0:
        total_time = time.time() - start_time
        print('Total time:', total_time)

    writefile.close()
    acc_writefile.close()
