import pytorch_lightning as pl

import torch
import torch.nn.functional as F


def distil_loss(student_loss, student_logits, teacher_logits, alpha=0.5, temperature=1.0):
    logits_loss = temperature**2 * F.kl_div(
        F.log_softmax(student_logits/temperature, dim=-1),
        F.softmax(teacher_logits/temperature, dim=-1),
        reduction='batchmean'
    )
    # return alpha * student_loss + (1. - alpha) * logits_loss
    return student_loss + alpha * logits_loss


class StudentModel(pl.LightningModule):
    def __init__(self, student_model, teacher_model, tokenizer, temperature, alpha, ids_to_clean_text, calculate_scores):
        super().__init__()
        self.teacher_model = teacher_model.eval()
        self.model = student_model
        self.tokenizer = tokenizer

        self.temperature = temperature
        self.alpha = alpha

        self.ids_to_clean_text = ids_to_clean_text
        self.calculate_scores = calculate_scores

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            teacher_outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                lm_labels=lm_labels,
                decoder_attention_mask=batch['target_mask']
            )

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        kd_loss = distil_loss(
            outputs.loss, outputs.logits, teacher_outputs.logits, alpha=self.alpha, temperature=self.temperature)

        return kd_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
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

        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        em_score = 0
        accuracy = 0

        em_score, accuracy = self.calculate_scores(preds, targets)

        em_score = torch.tensor(em_score, dtype=torch.float32)
        accuracy = torch.tensor(accuracy, dtype=torch.float32)

        self.log('em_score', em_score, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3)
        return [optimizer]
