# pylint: disable=import-error

import csv
import os
import time
impo
from transformers import T5Tokenizer
from torch.utils.data import DataLoader

from dataset import CKLDataset
from utils import load_dataset


def clean_up(text):
    return text.replace('<pad>', '').replace('</s>', '').replace(".", '') \
        .replace(',', '').replace("'", '').replace('"', '')


def evaluate(args, model_checkpoint, Model):
    model = Model(args)

    if args.checkpoint_path != "":
        if model_checkpoint is None:
            files = os.listdir(args.checkpoint_path)

            model_checkpoint = os.path.join(args.checkpoint_path, max(files, key=lambda x: int(
                x.split("step=")[1].split(".")[0])))

        model = Model.load_from_checkpoint(
            checkpoint_path=model_checkpoint, hparams=args, strict=False)

    model.eval()
    model.to('cuda')

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    if args.mode in ['pretrain', 'finetune']:
        stream_dataset, ids_to_answers = load_dataset('validation', args)
    else:
        raise Exception('Select the correct mode please.')

    total_cnt = 0
    em_correct_num = 0
    old_em_correct_num = 0
    new_em_correct_num = 0
    # accuracy_correct_num = 0

    # If folder doesn't exist, then create it.
    output_folder = ("/".join((args.output_log.split('/'))[:-1]))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    collector = []
    rows_to_write = []

    for idx, row in stream_dataset.iterrows():
        collector.append(row)
        if len(collector) >= args.eval_batch_size or idx == len(stream_dataset) - 1:

            loader = DataLoader(CKLDataset(collector, 'validation', tokenizer,
                                args), batch_size=len(collector), shuffle=False)

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
                texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
                targets = model.ids_to_clean_text(batch['target_ids'])

                for i in range(len(batch['source_ids'])):
                    total_cnt += 1
                    lines = clean_up(texts[i])
                    ground_truth = targets[i]
                    predicted = dec[i]
                    ids = batch['label_ids'][i].item()

                    if args.dataset == 'invariantlama':
                        em_correct_num += model.exact_match_score(
                            predicted, ground_truth)
                        rows_to_write.append(
                            [ids, lines, ground_truth, predicted])

                    elif args.dataset == 'updatedlama':
                        old_answer_list = ids_to_answers[str(ids)][0]['old']
                        new_answer_list = ids_to_answers[str(ids)][0]['new']
                        old_em_correct = False
                        new_em_correct = False
                        old_global_answer = old_answer_list[0]
                        new_global_answer = new_answer_list[0]

                        for answer in old_answer_list:
                            if model.exact_match_score(predicted, answer):
                                old_em_correct = True
                                old_global_answer = answer
                        if old_em_correct:
                            old_em_correct_num += 1

                        for answer in new_answer_list:
                            if model.exact_match_score(predicted, answer):
                                new_em_correct = True
                                new_global_answer = answer
                        if new_em_correct:
                            new_em_correct_num += 1

                        rows_to_write.append(
                            [ids, lines, old_global_answer, new_global_answer, predicted])

                    elif args.dataset in ['newlama', 'newlama_easy']:
                        answer_list = ids_to_answers[str(ids)]
                        em_correct = False
                        global_answer = answer_list[0]
                        for answer in answer_list:
                            if model.exact_match_score(predicted, answer):
                                em_correct = True
                                global_answer = answer
                        if em_correct:
                            em_correct_num += 1
                        rows_to_write.append(
                            [ids, lines, global_answer, predicted])
                    # elif args.dataset == 'WNED' or args.dataset == 'CWEB':
                    #     accuracy = model.accuracy_match_score(
                    #         predicted, ground_truth)
                    #     if accuracy == 1:
                    #         accuracy_correct_num += 1
                    #     rows_to_write.append([lines, ground_truth, predicted])
                    else:
                        raise NameError(
                            'Select the correct Dataset for zeroshot evaluation!')

                collector = []

    print(f'End time: {time.time() - start_time}')

    if args.dataset == 'updatedlama':
        rows_to_write.append(
            [old_em_correct_num, old_em_correct_num / total_cnt])
        rows_to_write.append(
            [new_em_correct_num, new_em_correct_num / total_cnt])

        print(
            f'Number of old correct predictions: {old_em_correct_num} / {total_cnt}. Percentage : {old_em_correct_num / total_cnt}')
        print(
            f'Number of new correct predictions: {new_em_correct_num} / {total_cnt}. Percentage : {new_em_correct_num / total_cnt}')
    # elif args.dataset == 'WNED' or args.dataset == 'CWEB':
    #     rows_to_write.append([accuracy_correct_num, accuracy_correct_num / total_cnt])
    #     print(
    #         f'Number of correct predictions: {accuracy_correct_num}. Percentage : {accuracy_correct_num / total_cnt}')
    else:
        rows_to_write.append([em_correct_num, em_correct_num / total_cnt])
        print(
            f'Number of correct predictions: {em_correct_num} / {total_cnt}. Percentage : {em_correct_num / total_cnt}')

    with open(args.output_log, 'w', newline='', encoding='utf-8') as writefile:
        writer = csv.writer(writefile)
        writer.writerows(rows_to_write)




def ACC():



def



