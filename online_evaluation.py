# pylint: disable=import-error

import time
import torch

from torch.utils.data import DataLoader
from fastdtw import fastdtw
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

from dataset import CKLDataset


def evaluate(args, model, df, tokenizer, rank):
    torch.cuda.empty_cache()

    model.to('cuda')
    model.eval()

    stream_datasets = df.groupby('date')
    def custom_sort(group):
        group['s_date'] = pd.to_datetime(group['date'])
        return group
    stream_datasets = stream_datasets.apply(custom_sort).reset_index(drop=True)
    stream_datasets = stream_datasets.groupby('s_date')

    metrics = []

    final_m_k = []
    final_w_k = []

    model_knowledge = []
    world_knowledge = []
    start_time = time.time()

    for date, stream_dataset in stream_datasets:
        total_cnt = 0
        em_correct_num = 0

        m_k = []
        w_k = []

        collector = []

        embedding_layer = model.model.get_input_embeddings()
        if rank == 0:
            print('Evaluating -', date)

        stream_dataset.reset_index(inplace=True)
        # stream_dataset = stream_dataset.sample(frac=0.5)
        # stream_dataset.reset_index(inplace=True)

        for idx, row in stream_dataset.iterrows():
            collector.append(row)
            if idx == len(stream_dataset) - 1:
            # if len(collector) >= args.eval_batch_size or idx == len(stream_dataset) - 1:
            #     # ====================== select random sample for evaluation ===========================
            #     # 计算要挑选的数据数量
            #     sample_size = int(len(collector) * 0.1)
            #     # 随机挑选指定数量的数据
            #     collector = random.sample(collector, sample_size)
            #     # ======================= select random sample for evaluation ==========================
                loader = DataLoader(CKLDataset(collector, 'test', tokenizer,
                                    args), batch_size=args.eval_batch_size, shuffle=False)

                for batch in iter(loader):
                    if 't5' in args.model_name_or_path:
                        outs = model.model.generate(
                            batch["source_ids"].cuda(),
                            attention_mask=batch["source_mask"].cuda(),
                            use_cache=True,
                            decoder_attention_mask=batch['target_mask'].cuda(),
                            max_length=args.max_output_length,
                            num_beams=2,
                            early_stopping=True,
                        )
                    else:
                        outs = model.model.generate(
                            batch["source_ids"].cuda(),
                            attention_mask=batch["source_mask"].cuda(),
                            pad_token_id=50256,
                            use_cache=True,
                            max_length=args.max_output_length+1,
                            num_beams=2,
                            early_stopping=True,
                        )

                    dec = model.ids_to_clean_text(outs)
                    targets = model.ids_to_clean_text(batch['target_ids'])

                    for i in range(len(batch['source_ids'])):
                        total_cnt += 1
                        ground_truth = targets[i]
                        predicted = dec[i]

                        em_correct_num += model.exact_match_score(
                            predicted, ground_truth)
                    # ------------------------ DTW ------------------------
                    # outputs = model.model(input_ids=batch["source_ids"].cuda(), decoder_input_ids=batch['target_ids'].cuda())
                    # p_emb = outputs.encoder_last_hidden_state
                    # p_emb = torch.mean(p_emb.detach(), dim=[1, 2]).cpu().numpy()
                    # embedding_layer = model.model.get_input_embeddings()
                    # g_emb = embedding_layer(batch['target_ids'].cuda())
                    # g_emb = torch.mean(g_emb.detach(), dim=[1, 2]).cpu().numpy()
                    # model_knowledge.extend(p_emb)
                    # world_knowledge.extend(g_emb)
                    lm_labels = batch['target_ids']
                    lm_labels[lm_labels[:, :] == -100] = 0

                    with torch.no_grad():
                        if 't5' in args.model_name_or_path:
                            outputs = model.model(
                                input_ids=batch['source_ids'].cuda(),
                                attention_mask=batch['source_mask'].cuda(),
                                labels=batch['target_ids'].cuda(),
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                output_hidden_states=True
                            )
                            p_emb = outputs.encoder_last_hidden_state
                        else:
                            outputs = model.model(
                                input_ids=batch['source_ids'].cuda(),
                                attention_mask=batch['source_mask'].cuda(),
                                labels=batch['target_ids'].cuda(),
                                output_hidden_states=True
                            )
                            p_emb = outputs.hidden_states[-1]
                        g_emb = embedding_layer(lm_labels.cuda())



                    # [B, N, L] --> [B, 1, 1] --> [B]
                    # print(torch.mean(p_emb, dim=[1, 2]).cpu().numpy())
                    m_k.extend(torch.mean(
                        p_emb, dim=[1, 2]).cpu().numpy())
                    w_k.extend(torch.mean(
                        g_emb, dim=[1, 2]).cpu().numpy())
                    # ------------------------ DTW ------------------------
                collector = []
        # for calculate the forgetting and updating rate
        model_knowledge.append(m_k)
        world_knowledge.append(w_k)

        # for calculate the final DTW
        final_m_k.extend(m_k) # 2019-1, 2019-2
        final_w_k.extend(w_k) # 2019-1, 2019-2

        metrics.append(100 * em_correct_num / total_cnt)
    dtw = np.linalg.norm(np.array(final_m_k) - np.array(final_w_k))
    # dtw, _ = fastdtw(np.array(final_m_k).reshape(len(final_m_k), 1),
    #                  np.array(final_w_k).reshape(len(final_w_k), 1),
    #                  dist=euclidean)
    knowledge = {"model":model_knowledge, "world":world_knowledge}
    model.train()

    return metrics, dtw, knowledge, time.time() - start_time
