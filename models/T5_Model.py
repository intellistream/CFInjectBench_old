# pylint: disable=import-error

import re
import random
import string
import torch
import numpy as np
import pytorch_lightning as pl

from collections import deque
from torch.utils.data import RandomSampler, random_split
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel
)
import pdb
from models.Modular_T5 import T5ForConditionalGeneration as T5_Modular
from models.Modular_Small_T5 import T5ForConditionalGeneration as T5_Modular_Small
from models.Kadapter_2_T5 import T5ForConditionalGeneration as T5_Kadapter2
from models.Kadapter_3_T5 import T5ForConditionalGeneration as T5_Kadapter3
from models.Lora_T5 import T5ForConditionalGeneration as T5_Lora
from models.RecAdam import RecAdam

from sklearn.metrics import pairwise_distances
import numpy as np
from models.Lossnet import LossNet

from dataset import CKLDataset

import numpy as np
from sklearn.metrics import pairwise_distances
import torch

class CoresetGreedy:
    def __init__(self, batch):

        self.all_pts = np.array(batch.detach().cpu().numpy())  # Use 'source_ids' from the batch dictionary


        self.dset_size = len(self.all_pts)  # Get the total number of points
        self.min_distances = np.inf * np.ones(self.dset_size)  # Initialize minimum distances to infinity
        self.already_selected = set()  # Initialize an empty set to keep track of selected points


    def update_distances(self, new_indices):
        # Calculate distances between new indices and all points and update the minimum distances accordingly
        dists = pairwise_distances(self.all_pts[new_indices], self.all_pts, metric='euclidean')
        self.min_distances = np.minimum(self.min_distances, np.min(dists, axis=0))

    def sample(self, sample_ratio):
        # Calculate sample size based on the input ratio
        sample_size = int(self.dset_size * sample_ratio)
        new_indices = []
        for _ in range(sample_size):
            if not self.already_selected:
                # If no points have been selected, choose one at random
                new_idx = np.random.choice(self.dset_size)
            else:
                # Otherwise, select the point that maximizes the minimum distance to already selected points
                # If the point has already been selected, it sets its minimum distance to zero and finds the next point with the maximum minimum distance
                new_idx = np.argmax(self.min_distances)
                while new_idx in self.already_selected:
                    self.min_distances[new_idx] = 0
                    new_idx = np.argmax(self.min_distances)

            self.already_selected.add(new_idx)  # Add the new index to the set of selected points
            self.update_distances([new_idx])  # Update the minimum distances with the new index
            new_indices.append(new_idx)  # Add the new index to the list of new indices

        return new_indices  # Return the list of new indices




class T5(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dataset = None

        self.mem_buff = deque(maxlen=10000)
        self.mem_ratio = 0.1
        self.epoch = 0
        self.coreset = hparams.coreset
        self.coreset_ratio = hparams.coreset_ratio
        self.repeat_num = hparams.repeat_num
        model_mapping = {
            'modular': T5_Modular,
            'modular_small': T5_Modular_Small,
            'kadapter2': T5_Kadapter2,
            'kadapter3': T5_Kadapter3,
            'lora': T5_Lora,
            'recadam': T5ForConditionalGeneration,
        }

        if hparams.method in model_mapping:
            if hparams.method == 'recadam':
                self.pretrained_model = model_mapping[hparams.method].from_pretrained(
                    hparams.model_name_or_path)
                self.freeze_params(self.pretrained_model)

            self.model = model_mapping[hparams.method].from_pretrained(
                hparams.model_name_or_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                hparams.model_name_or_path, output_hidden_states=True)

        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.model_name_or_path)

        # Freezing only encoder or the whole model
        if hparams.freeze_level == 0:  # Do not freeze any parameters
            # print('Not freezing any parameters!')
            pass
        elif hparams.freeze_level == 1:  # Freeze encoder only
            self.freeze_params(self.model.get_encoder())
        elif hparams.freeze_level == 2:  # Freeze encoder and decoder
            self.freeze_params(self.model)

        if hparams.coreset == "model":
            self.coreset_ratio = hparams.coreset_ratio
            self.lossnet = LossNet(depth=self.model.config.num_layers, input_dim=self.model.config.hidden_size)

        if hparams.method == 'modular_small':
            for name, param in self.model.named_parameters():
                if 'encoder_modular' in name:
                    param.requires_grad = True
        elif hparams.method in ['kadapter2', 'lora', 'kadapter3']:
            for name, param in self.model.named_parameters():
                if hparams.method in name:
                    param.requires_grad = True

        self.output_dir = self.hparams.output_dir

    def set_dataset(self, dataset):
        self.dataset = dataset
        if self.hparams.method == 'mixreview':
            self.dataset = self.set_memory_buffer()

        # train_size = int(0.8 * len(self.dataset))
        # val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = random_split(self.dataset, [0.8, 0.2])

    def set_memory_buffer(self):
        time_inv_relations = ['P19', 'P20', 'P279', 'P37', 'P449', 'P47', 'P138', 'P364', 'P527', 'P176', 'P27', 'P407', 'P30',
                              'P178', 'P1376', 'P131', 'P1412', 'P17', 'P276', 'P937', 'P140', 'P103', 'P190', 'P1001', 'P495', 'P36', 'P740', 'P361']

        temp = self.dataset.dataset[self.dataset.dataset['relation'].isin(
            time_inv_relations)].sample(frac=self.mem_ratio)

        if len(self.mem_buff) == 0:
            self.mem_buff.extend(temp.to_dict(orient='records'))
            return self.dataset

        self.mem_buff.extend(temp.to_dict(orient='records'))

        return ConcatDataset([self.dataset, CKLDataset(random.sample(self.mem_buff, min(len(self.mem_buff), int(self.mem_ratio * len(self.dataset)))), 'train', self.tokenizer, self.hparams, True)])

    def normalize_answer(self, s):
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
            text = text.replace('_X_', '')
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def accuracy_match_score(self, prediction, ground_truth):
        return int(prediction.strip() == ground_truth.strip())

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        accuracy = 0

        for idx, pred in enumerate(predictions):
            ground_truth = ground_truths[idx]
            em_score += self.exact_match_score(pred, ground_truth)
            accuracy += self.accuracy_match_score(pred, ground_truth)

        em_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score * 100, accuracy * 100

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        if self.coreset == 'model':
            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
            target_loss = []
            data_hidden_states = []
            for idx in range(len(batch[list(batch.keys())[0]])):
                outputs = self(
                    input_ids=batch["source_ids"][idx].unsqueeze(0),
                    attention_mask=batch["source_mask"][idx].unsqueeze(0),
                    lm_labels=lm_labels[idx].unsqueeze(0),
                    decoder_attention_mask=batch['target_mask'][idx].unsqueeze(0)
                )
                loss = outputs[0]
                target_loss.append(loss)
                data_hidden_states.append(outputs.encoder_hidden_states)

            hidden_states_data = [[] for _ in range(self.model.config.num_layers + 1)] # hidden states + last hidden state

            for hidden_states in data_hidden_states:
                for i, hidden_state in enumerate(hidden_states):
                    hidden_states_data[i].append(hidden_state.squeeze())
            for i, data in enumerate(hidden_states_data):
                hidden_states_data[i] = torch.stack(hidden_states_data[i])

            target_loss = torch.stack(target_loss)
            return target_loss, hidden_states_data
        else:
            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
            outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                lm_labels=lm_labels,
                decoder_attention_mask=batch['target_mask']
            )
            loss = outputs[0]
            return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch, batch_idx):

        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=10,
            num_beams=2,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])

        if self.coreset == 'model':
            loss, _ = self._step(batch)
            loss = torch.mean(loss)
        else:
            loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        em_score = 0
        accuracy = 0

        em_score, accuracy = self.calculate_scores(preds, targets)

        em_score = torch.tensor(em_score, dtype=torch.float32)
        accuracy = torch.tensor(accuracy, dtype=torch.float32)

        self.log('em_score', em_score, prog_bar=True, logger=True)

    def losspredloss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss



    def training_step(self, batch, batch_idx):
        for n in range(self.repeat_num):
            if self.coreset == 'model':
                self.model.eval()
                self.lossnet.eval()
                uncertainty = torch.tensor([]).cuda()
                num = int(len(batch[list(batch.keys())[0]]) * self.coreset_ratio)
                uncertainty = torch.tensor([]).cuda()
                with torch.no_grad():
                    lm_labels = batch["target_ids"]
                    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                    outputs = self(
                        input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        lm_labels=lm_labels,
                        decoder_attention_mask=batch['target_mask']
                    )
                    pred_loss = self.lossnet(
                        outputs.encoder_hidden_states)  # pred_loss = criterion(scores, labels) # ground truth loss
                    pred_loss = pred_loss.view(pred_loss.size(0))
                    uncertainty = torch.cat((uncertainty, pred_loss), 0)
                # Index in ascending order
                arg = np.argsort(uncertainty.cpu())[-num:]  # return the index starting from the smallest
                new_batch = {}
                for key in batch.keys():
                    new_batch[key] = [batch[key][i] for i in arg]
                    if key != 'date':
                        new_batch[key] = torch.stack(new_batch[key])
                self.model.train()
                self.lossnet.train()
                t_loss, hidden_states = self._step(new_batch)
                loss_logit = self.lossnet(hidden_states)
                loss_logit = loss_logit.view(loss_logit.size(0))
                predloss = self.losspredloss(loss_logit, t_loss)
                target_loss = torch.mean(t_loss)
                loss = target_loss + predloss
                self.log('target_loss', target_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('pred_loss', torch.mean(loss_logit), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('loss of pred_loss', predloss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            elif self.coreset == 'random':
                num = int(len(batch[list(batch.keys())[0]]) * self.coreset_ratio)
                idx = np.arange(len(list(batch.keys())[0]))
                random.shuffle(idx)
                arg = idx[:num]
                new_batch = {}
                for key in batch.keys():
                    new_batch[key] = [batch[key][i] for i in arg]
                    if key != 'date':
                        new_batch[key] = torch.stack(new_batch[key])
                loss = self._step(new_batch)

            elif self.coreset == 'K-center':
                embedding = self.model.get_input_embeddings()
                emb = embedding(batch['source_ids'].cuda()).mean(dim=1)
                coreset_greedy = CoresetGreedy(emb)
                selected_indices = coreset_greedy.sample(self.coreset_ratio)
                new_batch = {}
                for key in batch.keys():
                    new_batch[key] = [batch[key][i] for i in selected_indices]
                    if key != 'date':
                        new_batch[key] = torch.stack(new_batch[key])
                loss = self._step(new_batch)

            else:
                loss = self._step(batch)
                self.log("loss", loss)

            if n != self.repeat_num-1:
                # clear gradients
                self.optimizers().zero_grad()
                # backward
                loss.backward()
                # update parameters
                self.optimizers().step()
        return loss



    # def training_step(self, batch, batch_idx):
    #     if self.coreset == 'model':
    #         self.model.eval()
    #         self.lossnet.eval()
    #         uncertainty = torch.tensor([]).cuda()
    #         num = int(len(batch[list(batch.keys())[0]]) * self.coreset_ratio)
    #         uncertainty = torch.tensor([]).cuda()
    #         with torch.no_grad():
    #             lm_labels = batch["target_ids"]
    #             lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
    #             outputs = self(
    #                 input_ids=batch["source_ids"],
    #                 attention_mask=batch["source_mask"],
    #                 lm_labels=lm_labels,
    #                 decoder_attention_mask=batch['target_mask']
    #             )
    #             pred_loss = self.lossnet(
    #                 outputs.encoder_hidden_states)  # pred_loss = criterion(scores, labels) # ground truth loss
    #             pred_loss = pred_loss.view(pred_loss.size(0))
    #
    #             uncertainty = torch.cat((uncertainty, pred_loss), 0)
    #         # Index in ascending order
    #         arg = np.argsort(uncertainty.cpu())[-num:]  # return the index starting from the smallest
    #         new_batch = {}
    #         for key in batch.keys():
    #             new_batch[key] = [batch[key][i] for i in arg]
    #             if key != 'date':
    #                 new_batch[key] = torch.stack(new_batch[key])
    #         self.model.train()
    #         self.lossnet.train()
    #
    #         t_loss, hidden_states = self._step(new_batch)
    #         loss_logit = self.lossnet(hidden_states)
    #         loss_logit = loss_logit.view(loss_logit.size(0))
    #         predloss = self.losspredloss(loss_logit, t_loss)
    #         target_loss = torch.mean(t_loss)
    #         loss = target_loss + predloss
    #         self.log('target_loss', target_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #         self.log('pred_loss', torch.mean(loss_logit), on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #         self.log('loss of pred_loss', predloss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #
    #     elif self.coreset == 'random':
    #         num = int(len(list(batch.keys())[0]) * self.coreset_ratio)
    #         idx = np.arange(len(list(batch.keys())[0]))
    #         random.shuffle(idx)
    #         arg = idx[:num]
    #         new_batch = {}
    #         for key in batch.keys():
    #             new_batch[key] = [batch[key][i] for i in arg]
    #             if key != 'date':
    #                 new_batch[key] = torch.stack(new_batch[key])
    #
    #         loss = self._step(new_batch)
    #     elif self.coreset == 'K-center':
    #         embedding = self.model.get_input_embeddings()
    #         emb = embedding(batch['source_ids'].cuda()).mean(dim=1)
    #         coreset_greedy = CoresetGreedy(emb)
    #         selected_indices = coreset_greedy.sample(self.coreset_ratio)
    #         new_batch = {}
    #         for key in batch.keys():
    #             new_batch[key] = [batch[key][i] for i in selected_indices]
    #             if key != 'date':
    #                 new_batch[key] = torch.stack(new_batch[key])
    #
    #         loss = self._step(new_batch)
    #
    #
    #     else:
    #         loss = self._step(batch)
    #         self.log("loss", loss)
    #     return loss

    def on_train_epoch_start(self):
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method == 'recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 't5'
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0

            new_model = self.model
            pretrained_model = self.pretrained_model

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in new_model.named_parameters() if
                               not any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                               not any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                               any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": 0.0,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                               any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": 0.0,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type not in p_n]
                }
            ]
            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            if self.coreset == 'model':
                lossnet = self.lossnet
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.hparams.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in lossnet.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": 5e-4,
                    },
                    {
                        "params": [p for n, p in lossnet.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.hparams.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

        if self.hparams.use_lr_scheduling:
            denomniator = (self.hparams.n_gpu *
                           self.hparams.gradient_accumulation_steps) // 3
            steps_per_epoch = (len(self.dataset) // denomniator) + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch,
                                                               pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        return [optimizer]

    def train_dataloader(self):
        sampler = RandomSampler(self.train_set)
        dataloader = DataLoader(self.train_set, sampler=sampler,  batch_size=self.hparams.train_batch_size,
                                drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
