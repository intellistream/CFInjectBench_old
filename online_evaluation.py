import sys

from transformers import T5Tokenizer, GPT2Tokenizer
from Datasets import Pretrain
import torch
import time
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import csv
import os
import glob
from dtw import dtw_distance
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from utils import datetup2int
from torch.utils.data import SequentialSampler
from tqdm import tqdm


def traverse_directory(directory, extension='ckpt'):
    file_list = []
    pattern = os.path.join(directory, f'*.{extension}')
    for file_path in glob.glob(pattern):
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list

def clean_up(text):
    text = text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text
    # If folder doesn't exist, then create it.

def extract_step(filename):
    return int(filename.split('/')[-1].split('-')[-1].split('=')[-1].split('.')[0])

# def evaluate(rank, world_size, args, Model, result_list):
def evaluate(args, Model):
    model = Model(args)

    print(args.checkpoint_path)
    files = traverse_directory(args.checkpoint_path)

    ckpt_idx = 0
    dates = [x.split('/')[-1].split('=')[-1].split('.')[0] for x in files]
    dates = sorted(dates, key=lambda x: datetup2int(x))
    files = sorted(files, key=lambda x: datetup2int(x.split('/')[-1].split('=')[-1].split('.')[0]))
    print(dates)
    print(files)

    print('=' * 100)
    ckpt_path = files[ckpt_idx]
    # ckpt_path = files[ckpt_idx]
    print(ckpt_path)
    log_file = os.path.join(args.output_log, "test.csv")
    # log_file = os.path.join(args.output_log, "{}.csv".format(dates[ckpt_idx]))

    # MYDIR = ("/".join((log_file.split('/'))[:-1]))
    MYDIR = ("/".join((log_file.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    if ckpt_path != "":
        num_available_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0" if num_available_gpus > 0 else "cpu")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
        # model = Model.load_from_checkpoint(checkpoint_path=ckpt_path, hparams=args, strict=False)

    model.eval()

    model.to("cuda")
    # device = torch.device("cuda", local_rank)
    # model.to(device)

    # Wrap the model with DDP
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    # model = model.module

    total_cnt = 0
    em_correct_num = 0

    # Get Validation Data
    # Getting the Model type & Method
    if 't5' in args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    if 'gpt2' in args.model_name_or_path:
        tokenizer = model.tokenizer

    if args.mode == 'pretrain' or args.mode == 'finetune':
        dataset = Pretrain(tokenizer, 'test', None, input_length=args.max_input_length,
                           output_length=args.max_output_length, args=args)
        ids_to_answers = dataset.ids_to_answers
    else:
        raise Exception('Select the correct mode please.')
    print('Length of test data: ', len(dataset))
    sampler = SequentialSampler(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32, shuffle=False, num_workers=4)
    # loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, shuffle=False)
    # test
    # idx= 0
    # with torch.no_grad():
    #     for batch in tqdm(iter(loader), desc="check dataloader"):
    #         pass

    writefile = open(log_file, 'w', newline='')
    writer = csv.writer(writefile)
    model_knowledge = []
    world_knowledge = []
    all_acc = []
    # switch to next date ckpt
    next_date = dates[ckpt_idx + 1]

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(iter(loader), desc="Evaluate QA stream"):
            if batch['date'][0] == next_date:
                # record related metrics
                dtw_res, _ = fastdtw(np.array(model_knowledge).reshape(len(model_knowledge), 1),
                                     np.array(world_knowledge).reshape(len(world_knowledge), 1),
                                     dist=euclidean)
                model_knowledge = []
                world_knowledge = []
                acc = em_correct_num / total_cnt
                all_acc.append(acc)
                if ckpt_idx > 0:
                    bwt = acc - pre_acc
                else:
                    bwt = None
                print(f"\n>>> For model in {dates[ckpt_idx]}: BWT: {bwt}\tDTW: {dtw_res}\tACC: {acc}")
                writer.writerow([dates[ckpt_idx], dtw_res, bwt, acc])

                # load previous date data for BWT
                dataset = Pretrain(tokenizer, 'test', None, input_length=args.max_input_length,
                                   output_length=args.max_output_length, args=args, date=dates[ckpt_idx])
                sampler = SequentialSampler(dataset)
                pre_loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, shuffle=False)
                pre_total_cnt = 0
                pre_em_correct_num = 0

                # check next date
                if (ckpt_idx + 1) < len(dates):
                    ckpt_idx += 1
                    next_date = dates[ckpt_idx]
                    print("Next date is:", next_date)
                else:
                    next_date = None

                # load new ckpt to model
                ckpt_path = files[ckpt_idx]
                num_available_gpus = torch.cuda.device_count()
                device = torch.device("cuda:0" if num_available_gpus > 0 else "cpu")
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.eval()
                model.to("cuda")

                for pre_batch in tqdm(iter(pre_loader), desc="Evaluate previous stream"):
                    outs = model.model.generate(
                        pre_batch["source_ids"].to("cuda"),
                        attention_mask=pre_batch["source_mask"].to("cuda"),
                        use_cache=True,
                        decoder_attention_mask=None,
                        max_length=args.max_output_length,
                        num_beams=2,
                        early_stopping=True,
                    )
                    dec = model.ids_to_clean_text(outs)
                    targets = model.ids_to_clean_text(pre_batch['target_ids'])
                    pre_em_correct_num += model.exact_match_score(dec, targets)
                    pre_total_cnt += args.eval_batch_size
                    # ------------------------ DTW ------------------------
                    outputs = model.model(batch["source_ids"].to("cuda"))
                    p_emb = outputs.hidden_states[-1]
                    p_emb = torch.mean(p_emb, dim=[1, 2]).cpu().numpy()
                    embedding_layer = model.model.get_input_embeddings()
                    g_emb = embedding_layer(batch['target_ids'].cuda())
                    g_emb = torch.mean(g_emb, dim=[1, 2]).cpu().numpy()
                    model_knowledge.extend(p_emb)
                    world_knowledge.extend(g_emb)
                    # ------------------------ DTW ------------------------

                pre_acc = pre_em_correct_num / pre_total_cnt

                em_correct_num = 0
                total_cnt = 0

            outs = model.model.generate(
                batch["source_ids"].to("cuda"),
                attention_mask=batch["source_mask"].to("cuda"),
                use_cache=True,
                decoder_attention_mask=None,
                max_length=args.max_output_length,
                num_beams=2,
                early_stopping=True,
            )
            dec = model.ids_to_clean_text(outs)
            # texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = model.ids_to_clean_text(batch['target_ids'])
            em_correct_num += model.exact_match_score(dec, targets)

            # ------------------------ DTW ------------------------
            outputs         = model.model(batch["source_ids"].to("cuda"))
            p_emb           = outputs.hidden_states[-1]
            p_emb           = torch.mean(p_emb, dim=[1, 2]).cpu().numpy()
            embedding_layer = model.model.get_input_embeddings()
            g_emb           = embedding_layer(batch['target_ids'].cuda())
            g_emb           = torch.mean(g_emb, dim=[1, 2]).cpu().numpy()
            model_knowledge.extend(p_emb)
            world_knowledge.extend(g_emb)
            # ------------------------ DTW ------------------------

            total_cnt += args.eval_batch_size

    # dtw_res = dtw_distance(model_knowledge, world_knowledge)
    # dtw_res, _ = fastdtw(model_knowledge, world_knowledge, dist=euclidean)
    # print("DTW distance:", dtw_res)
    run_time = time.time() - start_time
    writefile.close()
    print(f'Running time: {run_time}')
    with open(log_file, 'a', newline='') as writefile:
        writer = csv.writer(writefile)
        writer.writerow([em_correct_num, em_correct_num / total_cnt])
    # print(f'Number of correct predictions: {em_correct_num}. Percentage : {em_correct_num / total_cnt}')

    print(f'Averaged accuracy of all step is: {sum(all_acc)/len(all_acc)}')
