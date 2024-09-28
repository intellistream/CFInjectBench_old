import pytorch_lightning as pl
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, Adafactor
from models.RecAdam import RecAdam
from models.Kadapter_Llama import LlamaWithKadapter as Llama_Kadapter
from peft import get_peft_model, LoraConfig, TaskType
import torch
import random
from collections import deque
from torch.utils.data import SequentialSampler, RandomSampler, random_split
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CKLDataset
import re
import string


class Llama(pl.LightningModule):
    def __init__(self, hparams):
        super(Llama, self).__init__()
        self.save_hyperparameters(hparams)

        self.save_hyperparameters(hparams)
        self.dataset = None

        self.mem_buff = deque(maxlen=10000)
        self.mem_ratio = 0.1
        self.epoch = 0
        self.coreset = hparams.coreset
        self.coreset_ratio = hparams.coreset_ratio
        self.repeat_num = hparams.repeat_num
        self.teacher_model = None

        if hparams.method=='kadapter':
            self.model = Llama_Kadapter.from_pretrained(hparams.model_name_or_path, torch_dtype=torch.float16,)
        elif hparams.method=='lora':
            self.model = LlamaForCausalLM.from_pretrained(hparams.model_name_or_path, torch_dtype=torch.float16)
            lora_config = LoraConfig(
                r=4,  # LoRA rank
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
        elif hparams.method=='recadam':
            self.model = LlamaForCausalLM.from_pretrained(hparams.model_name_or_path, torch_dtype=torch.float16)
            self.pretrained_model = LlamaForCausalLM.from_pretrained(hparams.model_name_or_path, torch_dtype=torch.float16)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        else:
            self.model = LlamaForCausalLM.from_pretrained(hparams.model_name_or_path, torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained(hparams.model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"  # Allow batched inference
        if hparams.freeze_level == 0:  # Do not freeze any parameters
            print('Not freezing any parameters!')
        elif hparams.freeze_level == 1:  # Freeze model
            self.freeze_params(self.model)

        if hparams.method == 'kadapter':
            for name, param in self.model.named_parameters():
                if 'kadapter' in name or 'lm_head' in name:
                    param.requires_grad = True
        elif hparams.method == 'lora':
            for name, param in self.model.named_parameters():
                if 'lora' in name or 'lm_head' in name:
                    param.requires_grad = True

        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

        self.output_dir = self.hparams.output_dir


    def set_dataset(self, dataset, kd=False):
        self.dataset = dataset
        if kd or self.hparams.method == 'mixreview':
            self.dataset = self.set_memory_buffer()

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
            # 移除 LLaMA 可能使用的特殊标记
            text = re.sub(r'<\|.+?\|>', '', text)  # 移除可能的控制标记
            text = text.replace('</s>', '')  # 移除结束标记
            text = text.replace('<s>', '')  # 移除开始标记
            text = text.replace('<unk>', '')  # 移除未知标记
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

    def generate_prompt(self, input=None):
        return f"""Write a response that appropriately answers the question. 
                    
                    ### Question:
                    {input}
                    
                    ### Response:
                    """

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,)

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
        )
        loss = outputs.loss
        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch, batch_idx):
        input_length = self.hparams.max_input_length
        max_length = self.hparams.max_output_length + 1

        # questions = self.tokenizer.batch_decode(batch["source_ids"])
        # inputs = [self.generate_prompt(q) for q in questions]
        # inputs = self.tokenizer(inputs, return_tensors="pt")
        # input_ids = inputs["input_ids"]

        # LLaMA specific generation config
        # generation_config = GenerationConfig(
        #     temperature=0.1,
        #     top_p=0.75,
        #     top_k=40,
        #     num_beams=4,
        #     max_tokens=max_length,
        #     # early_stopping=True,
        #     # pad_token_id=self.tokenizer.pad_token_id,
        #     # eos_token_id=self.tokenizer.eos_token_id,
        # )
        with torch.no_grad():
            generation_output = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                max_length=max_length,
                num_beams=2,
                early_stopping=True,
            )
        preds = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # preds =  [output.split("Response:")[1].strip() for output in preds]
        # source_texts = self.tokenizer.batch_decode(batch["source_ids"], skip_special_tokens=True,
        #                                            clean_up_tokenization_spaces=True)
        target_texts = self.tokenizer.batch_decode(batch["target_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # loss = self._step(batch)
        #
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        em_score, accuracy = self.calculate_scores(preds, target_texts)

        em_score = torch.tensor(em_score, dtype=torch.float32)

        self.log('em_score', em_score, prog_bar=True, logger=True)
        self.log('accuracy', accuracy, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        # self.model.train()
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def on_train_epoch_start(self):
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method == 'recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 'LlamaModel'  # 更改为 LLaMA 的模型类型
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0
            new_model = self.model
            pretrained_model = self.pretrained_model

            # LLaMA 模型的参数分组
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

            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False,
                                  relative_step=False)

        self.optimizer = optimizer
        # len_data = len(self.train_dataloader())
        # denomniator = self.hparams.n_gpu * self.hparams.gradient_accumulation_steps
        # steps_per_epoch = (len_data // denomniator) + 1
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
        #                                                    steps_per_epoch=steps_per_epoch, pct_start=0.1,
        #                                                    epochs=self.hparams.num_train_epochs,
        #                                                    anneal_strategy='linear', cycle_momentum=False)
        #
        # if self.hparams.use_lr_scheduling:
        #     return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        # else:
        return [optimizer]

    def train_dataloader(self):
        sampler = RandomSampler(self.train_set)
        dataloader = DataLoader(self.train_set, sampler=sampler, batch_size=self.hparams.train_batch_size,
                                drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers,
                          shuffle=False)

    def on_train_start(self):
        if self.teacher_model:
            self.model = self.teacher_model