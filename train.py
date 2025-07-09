import os
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

# from torch.nn.attention import SDPBackend, sdpa_kernel

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset


class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)
     
    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def run(data_path, log_path, mode, use_text):
    cad_recode_path = os.path.join(data_path, 'text2cad')
    train_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split='train',
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode)
    batch_size = 15
    accumulation_steps = 2

    if use_text:
        text_dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='train')
        train_dataset = ConcatDataset([train_dataset, text_dataset])
        batch_size = 8
        accumulation_steps = 4
  
    eval_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split='val',
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=None,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode)
    
    processor = AutoProcessor.from_pretrained(
        # Uncomment the following line to use a different model
        # "Qwen/Qwen2.5-VL-32B-Instruct",
        # "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        padding_side='left')
    model = Cadrille.from_pretrained(
        # Uncomment the following line to use a different model
        # "Qwen/Qwen2.5-VL-32B-Instruct",
        # "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa' # comment when using torch.nn.attention.SDPBackend.MATH
    )
    # class SDPAWrapper(torch.nn.Module):
    #     def __init__(self, model, backend=SDPBackend.MATH):  # or FLASH_ATTENTION
    #         super().__init__()
    #         self.model = model
    #         self.backend = backend

    #     def forward(self, *args, **kwargs):
    #         with sdpa_kernel(self.backend):
    #             return self.model(*args, **kwargs)
    trainer = Trainer(
        # model=SDPAWrapper(model),
        model=model,
        args=TrainingArguments(
            output_dir=log_path,
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=18,
            max_steps=120000,
            lr_scheduler_type='cosine',
            learning_rate=2e-4,
            warmup_steps=1000,
            weight_decay=0.01,
            gradient_accumulation_steps=accumulation_steps,
            remove_unused_columns=False,
            logging_steps=1000,
            save_total_limit=2,
            save_strategy='steps',
            save_steps=10000,
            eval_strategy='steps',
            eval_steps=10000,
            load_best_model_at_end=True,         
            report_to=None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate, processor=processor, n_points=256),
        tokenizer=processor,
        callbacks=[PrintToFileCallback()])
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--log-path', type=str, default='./work_dirs')
    parser.add_argument('--mode', type=str, default='pc_img')
    parser.add_argument('--use-text', action='store_true')

    args = parser.parse_args()
    run(args.data_path, args.log_path, args.mode, args.use_text)
