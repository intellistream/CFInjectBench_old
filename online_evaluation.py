# pylint: disable=import-error

import time
import torch

from torch.utils.data import DataLoader
from fastdtw import fastdtw
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean

from dataset import CKLDataset


def evaluate(args, model, df, tokenizer):
    torch.cuda.empty_cache()

    model.to('cuda')
    model.eval()

    stream_datasets = df.groupby('date')
    metrics = []
    dtw = 0

    start_time = time.time()

    for date, stream_dataset in stream_datasets:
        total_cnt = 0
        em_correct_num = 0

        model_knowledge = []
        world_knowledge = []
        collector = []

        print('Evaluating -', date)

        stream_dataset.reset_index(inplace=True)
        # stream_dataset = stream_dataset.sample(frac=0.5)
        # stream_dataset.reset_index(inplace=True)

        for idx, row in stream_dataset.iterrows():
            collector.append(row)
            if idx == len(stream_dataset) - 1:
            # if len(collector) >= args.eval_batch_size or idx == len(stream_dataset) - 1:
                loader = DataLoader(CKLDataset(collector, 'test', tokenizer,
                                    args), batch_size=args.eval_batch_size, shuffle=False)

                for batch in tqdm(iter(loader), desc="Evaluate"):
                    outs = model.model.generate(
                        batch["source_ids"].cuda(),
                        attention_mask=batch["source_mask"].cuda(),
                        use_cache=True,
                        decoder_attention_mask=batch['target_mask'].cuda(),
                        max_length=args.max_output_length,
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
                    outputs = model.model(input_ids=batch["source_ids"].cuda(), decoder_input_ids=batch['target_ids'].cuda())
                    p_emb = outputs.encoder_last_hidden_state
                    p_emb = torch.mean(p_emb.detach(), dim=[1, 2]).cpu().numpy()
                    embedding_layer = model.model.get_input_embeddings()
                    g_emb = embedding_layer(batch['target_ids'].cuda())
                    g_emb = torch.mean(g_emb.detach(), dim=[1, 2]).cpu().numpy()
                    model_knowledge.extend(p_emb)
                    world_knowledge.extend(g_emb)
                    # ------------------------ DTW ------------------------

                collector = []
        dtw, _ = fastdtw(np.array(model_knowledge).reshape(len(model_knowledge), 1),
                             np.array(world_knowledge).reshape(len(world_knowledge), 1),
                             dist=euclidean)
        metrics.append(100 * em_correct_num / total_cnt)

    model.train()

    return metrics, dtw, time.time() - start_time
