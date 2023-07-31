# pylint: disable=import-error
import os
import time
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import T5Tokenizer, GPT2Tokenizer

from utils import load_dataset

class CustomModelCheckpoint(pl.Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.pre_date = None
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            # self.trainer.logger.log_metrics(f"Folder '{dirpath}' created successfully.")
            # print(f"Folder '{dirpath}' already exists.")
        self.save_path = dirpath + "/epoch={epoch}-date={date}.ckpt"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # 获取当前批次的输入数据和对应标签
        # # for batch in iter(trainer.dataloader):
        # for i in range(len(batch['date'])):
        #     print(batch['date'][i])
        date = batch["date"][0]
        # print(date)
        if self.pre_date is None:
            self.pre_date = batch["date"][-1]
        if self.pre_date != date:
            torch.save(pl_module.state_dict(), self.save_path.format(epoch=trainer.current_epoch, date=self.pre_date))
            if trainer.global_rank == 0:
                print(f"\nSave model in {self.save_path.format(epoch=trainer.current_epoch, date=self.pre_date)}")
            self.pre_date = batch["date"][-1]
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.save(pl_module.state_dict(), self.save_path.format(epoch=trainer.current_epoch, date=self.pre_date))
        if trainer.global_rank == 0:
            print(f"\nSave model in {self.save_path.format(epoch=trainer.current_epoch, date=self.pre_date)}")



def train(args, Model, wandb_logger):
    if 't5' in args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    model = Model(args)

    start_time = time.time()

    collector = []

    trainer = pl.Trainer(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins='deepspeed_stage_2' if args.use_deepspeed else [],
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.use_deepspeed else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=True,
        val_check_interval=args.val_check_interval,
        callbacks=[CustomModelCheckpoint(dirpath=args.output_dir)],
        accelerator=args.accelerator,
        logger=wandb_logger,
    )

    trainer.fit(model)

    # for idx, line in enumerate(stream_dataset):
    #     collector.append(line)
    #     if (idx + 1) % args.stream_mini_batch_size == 0 or idx == len(stream_dataset) - 1:
    #         if 'gpt2' in args.model_name_or_path:
    #             model.set_dataset(CKLDataset(collector, 'train', model.tokenizer, args))
    #         else:
    #             model.set_dataset(CKLDataset(collector, 'train', tokenizer, model, args))
    #         trainer.fit(model)
    #         collector = []
    #
    #         if idx == len(stream_dataset) - 1:
    #             trainer.max_epochs -= 1


    print('Total time:', time.time() - start_time)
