# pylint: disable=import-error

import time
import torch

from torch.utils.data import DataLoader

from dataset import CKLDataset


def evaluate(args, model, df, tokenizer):
    torch.cuda.empty_cache()

    model.to('cuda')
    model.eval()

    stream_datasets = df.groupby('date')
    metrics = []

    for date, stream_dataset in stream_datasets:
        total_cnt = 0
        em_correct_num = 0

        collector = []

        print('Evaluating -', date)
        start_time = time.time()

        stream_dataset.reset_index(inplace=True)
        # stream_dataset = stream_dataset.sample(frac=0.5)
        # stream_dataset.reset_index(inplace=True)

        for idx, row in stream_dataset.iterrows():
            collector.append(row)
            if len(collector) >= args.eval_batch_size or idx == len(stream_dataset) - 1:
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
                collector = []

        metrics.append(100 * em_correct_num / total_cnt)

    model.train()

    return metrics,  time.time() - start_time
