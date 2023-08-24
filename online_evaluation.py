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
    model_knowledge = []
    world_knowledge = []
    start_time = time.time()

    for date, stream_dataset in stream_datasets:
        total_cnt = 0
        em_correct_num = 0


        collector = []

        embedding_layer = model.model.get_input_embeddings()
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

                for batch in iter(loader):
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
                    lm_labels = batch['target_ids']
                    lm_labels[lm_labels[:, :] == -100] = 0

                    with torch.no_grad():
                        outputs = model.model(input_ids=batch['source_ids'].cuda(),
                                              attention_mask=batch['source_mask'].cuda(),
                                              labels=batch['target_ids'].cuda(),
                                              decoder_attention_mask=batch['target_mask'].cuda(),
                                              output_hidden_states=True)
                        g_emb = embedding_layer(lm_labels.cuda())
                    p_emb = outputs.encoder_last_hidden_state

                    model_knowledge.extend(torch.mean(
                        p_emb, dim=[1, 2]).cpu().numpy())
                    world_knowledge.extend(torch.mean(
                        g_emb, dim=[1, 2]).cpu().numpy())
                    # ------------------------ DTW ------------------------

                collector = []
        metrics.append(100 * em_correct_num / total_cnt)
    dtw, _ = fastdtw(np.array(model_knowledge).reshape(len(model_knowledge), 1),
                     np.array(world_knowledge).reshape(len(world_knowledge), 1),
                     dist=euclidean)
    model.train()

    return metrics, dtw, time.time() - start_time
