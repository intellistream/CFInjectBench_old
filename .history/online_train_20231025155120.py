# pylint: disable=import-error

import os
import csv
import shutil
import time
import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


from transformers import T5Tokenizer, GPT2Tokenizer
from collections import deque

from dataset import CKLDataset
from online_evaluation import evaluate
from utils import load_dataset
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from copy import deepcopy
from copy import deepcopy

def get_dtw(m, w):
    # # dtw, _ = fastdtw(np.array(m).reshape(len(m), 1),
    # #                np.array(w).reshape(len(w), 1),
    # #                dist=euclidean)
    dtw = np.linalg.norm(np.array(m) - np.array(w))
    dtw = np.linalg.norm(np.array(m) - np.array(w))
    return dtw

def find_next_entry(start_idx, train_stream_df):
    next_entry = None
    for idx, row in train_stream_df.iterrows():
        if idx >= start_idx:
            if next_entry and next_entry != row['date'] or idx == len(train_stream_df) - 1:
                return next_entry
            next_entry = row['date']
    return None

class CustomModelCheckpoint(pl.Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.pre_date = None
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.save_path = dirpath + "/date={date}.ckpt"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.pre_date = batch["date"][0]

    def on_train_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict(), self.save_path.format(date=self.pre_date))
        if trainer.global_rank == 0:
            print(f"\nSave model in {self.save_path.format(date=self.pre_date)}")

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
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu',
        enable_progress_bar=False,
        max_epochs=args.num_train_epochs,
        precision=16 if args.use_deepspeed else 32,
        devices=args.n_gpu,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        # # callbacks=[CustomModelCheckpoint(dirpath=args.output_dir)],
        strategy='ddp'

    )

    last_entry = None 

    bwt = []
    fwt = []
    acc = []
    dtw = []
    pre_metric = []
    knowledge = []
    eval_time = []

    periods = deque(maxlen=3)
    first_time = True

    writefile = open(f'{args.output_log}results.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(writefile)
    writer.writerow(["Date", "EM", "BWT", "FWT", "DTW", "Forget", "Update", "Time"])
    writefile.flush()
    flag = True
    flag = True

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
                print(f"Coreset method: {args.coreset}")
                print(f"Coreset ratio: {args.coreset_ratio}")
                start_train = time.time()
            if args.method != 'initial':
                if args.method != 'initial':
                trainer.fit(model)
                    trainer.fit_loop.max_epochs += args.num_train_epochs
            if trainer.global_rank == 0:
                train_time = time.time() - start_train
                print(f'TRAIN TIME:{train_time}')

            collector = []
            if first_time:
                periods.append(last_entry)
            next_entry = find_next_entry(idx, train_stream_df)
            # for the end of the train stream, periods = ['date(t-1)', 'date(t)', None]
            periods.append(next_entry)

            if trainer.global_rank == 0:
                print(f"For periods: {periods}")
                metrics, dtw_res, k, e_time = evaluate(
                    args, model, test_stream_df[test_stream_df['date'].isin(periods)], tokenizer, trainer.global_rank)



                if len(periods) == 3:
                    # metric, knowledge:
                    # 1-1 1-2
                    # 2-1 2-2 2-3
                    # ... 3-2 3-3 3-4
                    # ... ... 4-3 4-4
                    if len(knowledge['model']) == 2:
                        forget = get_dtw(k['model'][0], knowledge['model'][0])
                        update = get_dtw(k['model'][1], knowledge['model'][1])
                        bwt_res = metrics[0] - pre_metric[0]
                        fwt_res = metrics[1] - pre_metric[1]
                        bwt_res = metrics[0] - pre_metric[0]
                        fwt_res = metrics[1] - pre_metric[1]

                    if len(knowledge['model']) == 3:
                        forget = get_dtw(k['model'][0], knowledge['model'][1])
                        update = get_dtw(k['model'][1], knowledge['model'][2])
                        bwt_res = metrics[0] - pre_metric[1]
                        fwt_res = metrics[1] - pre_metric[2]
                        bwt_res = metrics[0] - pre_metric[1]
                        fwt_res = metrics[1] - pre_metric[2]

                    # bwt.append(bwt_res)
                    # fwt.append(fwt_res)
                    # bwt.append(bwt_res)
                    # fwt.append(fwt_res)

                    print("Forget:", forget)
                    print("Update:", update)
                    print('BWT:', bwt_res)
                    print('FWT:', fwt_res)
                    print('BWT:', bwt_res)
                    print('FWT:', fwt_res)

                # dtw.append(dtw_res)
                knowledge = deepcopy(k)
                # dtw.append(dtw_res)
                knowledge = deepcopy(k)
                pre_metric = metrics
                if len(metrics) == 2:
                    acc_idx = 0
                else:
                    acc_idx = 1
                # acc.append(metrics[acc_idx])
                # acc.append(metrics[acc_idx])
                eval_time.append(e_time)
                print('DTW:', dtw_res)
                print('ACC:', metrics[acc_idx])
                print('DTW:', dtw_res)
                print('ACC:', metrics[acc_idx])
                print('TIME:', eval_time[-1])

                writer = csv.writer(writefile)
                # writer.writerow(["Date", "EM", "BWT", "FWT", "DTW", "Forget", "Update", "Time"])
                if first_time:
                    writer.writerow([periods[0], metrics[acc_idx], None, None, dtw_res, None, None, train_time])
                    writer.writerow([periods[0], metrics[acc_idx], None, None, dtw_res, None, None, train_time])
                    first_time = False
                    writefile.flush()
                    writefile.flush()
                else:
                    writer.writerow([periods[1], metrics[acc_idx], bwt_res, fwt_res, dtw_res, forget, update, train_time])
                    # writer.writerow([periods[1], acc[-1], bwt[-1], fwt[-1], dtw[-1], forget, update, train_time])
                    writer.writerow([periods[1], metrics[acc_idx], bwt_res, fwt_res, dtw_res, forget, update, train_time])
                    # writer.writerow([periods[1], acc[-1], bwt[-1], fwt[-1], dtw[-1], forget, update, train_time])
                    writefile.flush()

            trainer.strategy.barrier()

        # #============== control the start date =============
        # if row['date'] == '2019-8':
        #     flag = False
        # if flag:
        #     continue
        collector.append(row.to_dict())
        last_entry = row['date']

    writefile.close()
    writefile.close()
    trainer.strategy.barrier()


    # if trainer.global_rank == 0:
    #     total_time = time.time() - start_time
    #     print('Total time:', total_time)
    #
    #     with open(f'{args.output_log}all.csv', 'w', newline='', encoding='utf-8') as writefile:
    #         writer = csv.writer(writefile)
    #         writer.writerows([acc,[0] + bwt, eval_time])
    #
    #     with open(f'{args.output_log}results.csv', 'w', newline='', encoding='utf-8') as writefile:
    #         writer = csv.writer(writefile)
    #         writer.writerow(['ACC', 'BWT', 'DTW', 'TIME', 'TRAIN_TIME'])
    #         writer.writerow([sum(acc)/len(acc), sum(bwt)/len(bwt), sum(dtw)/len(dtw), total_time, total_time-sum(eval_time)])