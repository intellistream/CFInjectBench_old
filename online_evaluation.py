# pylint: disable=import-error

import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import re
import string

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

                for batch in tqdm(loader, total=len(loader)):
                    if 't5' in args.model_name_or_path:
                        with torch.no_grad():
                            outs = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                max_length=args.max_output_length,
                                num_beams=2,
                                early_stopping=True,
                            )
                    elif 'llama' in args.model_name_or_path:
                        # from transformers import GenerationConfig
                        # generation_config = GenerationConfig(
                        #     temperature=0.1,
                        #     top_p=0.75,
                        #     top_k=40,
                        #     num_beams=4,
                        #     max_tokens=18,
                        #     # early_stopping=True,
                        #     # pad_token_id=self.tokenizer.pad_token_id,
                        #     # eos_token_id=self.tokenizer.eos_token_id,
                        # )
                        # with torch.no_grad():
                        #     outs = model.model.generate(
                        #         input_ids=batch["source_ids"].cuda(),
                        #         # generation_config=generation_config,
                        #         max_tokens=18,
                        #         # max_new_tokens=18,
                        #         return_dict_in_generate=False,
                        #         output_scores=True,
                        #         use_cache=False,
                        #     )
                        with torch.no_grad():
                            # outs = model.model.generate(
                            #     batch["source_ids"].cuda(),
                            #     attention_mask=batch["source_mask"].cuda(),
                            #     use_cache=False,
                            #     max_new_tokens=args.max_output_length,
                            #     num_beams=2,
                            #     early_stopping=True,
                            # )
                            outs = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                max_length=args.max_output_length + 1,
                                num_beams=2,
                                early_stopping=True,
                            )
                    else:
                        with torch.no_grad():
                            outs = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                pad_token_id=50256,
                                use_cache=True,
                                max_length=args.max_output_length+1,
                                num_beams=2,
                                early_stopping=True,
                            )

                    if hasattr(model, 'ids_to_clean_text'):
                        dec = model.ids_to_clean_text(outs)
                        targets = model.ids_to_clean_text(batch['target_ids'])
                    else:
                        dec = ids_to_clean_text(tokenizer, outs)
                        targets = ids_to_clean_text(tokenizer, batch['target_ids'])

                    for i in range(len(batch['source_ids'])):
                        total_cnt += 1
                        ground_truth = targets[i]
                        predicted = dec[i]

                        if hasattr(model, 'exact_match_score'):
                            em_correct_num += model.exact_match_score(predicted, ground_truth)
                        else:
                            em_correct_num += exact_match_score(predicted, ground_truth)

        metrics.append(100 * em_correct_num / total_cnt)
    """
    dtw = np.linalg.norm(np.array(final_m_k) - np.array(final_w_k))
    # dtw, _ = fastdtw(np.array(final_m_k).reshape(len(final_m_k), 1),
    #                  np.array(final_w_k).reshape(len(final_w_k), 1),
    #                  dist=euclidean)
    knowledge = {"model":model_knowledge, "world":world_knowledge}
    """
    model.train()

    return metrics, time.time() - start_time

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return list(map(str.strip, gen_text))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def rid_of_specials(text):
        # 移除 LLaMA 可能使用的特殊标记
        text = re.sub(r'<\|.+?\|>', '', text)  # 移除可能的控制标记
        text = text.replace('</s>', '')  # 移除结束标记
        text = text.replace('<s>', '')  # 移除开始标记
        text = text.replace('<unk>', '')  # 移除未知标记
        return text

    return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))