# basically the same as old train.py from 7/16/2025 (check colab)
# adapted claude feedback for hyperparameters
import os
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset

# from torch.nn.attention import sdpa_kernel, SDPBackend # remove when using transformers sdpa

class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)
     
    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')

class ClearCUDACacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def run(data_path, log_path, mode, use_text):
    cad_recode_path = os.path.join(data_path, 'cad-recode-v1.5')
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
    batch_size = 4
    accumulation_steps = 2

    if use_text:
        text_dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='train')
        train_dataset = ConcatDataset([train_dataset, text_dataset])
        batch_size = 4
        accumulation_steps = 2
  
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
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        padding_side='left',
        use_fast=True)
    model = Cadrille.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",

        # the following two lines are NOT to be used together; either use one or the other
        # torch_dtype=torch.bfloat16, # handling dtype in TrainingArguments
        attn_implementation='sdpa' # comment when using torch.nn.attention.SDPBackend.MATH
    )
    # comment this block when using transformers sdpa
    # class SDPAWrapper(torch.nn.Module):
    #     def __init__(self, model, backend=SDPBackend.FLASH_ATTENTION):  # or FLASH_ATTENTION
    #         super().__init__()
    #         self.model = model
    #         self.backend = backend

    #     def forward(self, *args, **kwargs):
    #         with sdpa_kernel(self.backend):
    #             return self.model(*args, **kwargs)
    trainer = Trainer(
        # model=SDPAWrapper(model), # wrapping the model to use SDPA
        model=model,
        args=TrainingArguments(
            output_dir=log_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,  # Same as train for consistency
    
            # LEARNING RATE SCHEDULE - Key for preventing catastrophic forgetting
            learning_rate=3e-5,  # Much lower than your 2e-4
            lr_scheduler_type='cosine_with_restarts',  # Better than plain cosine
            warmup_steps=1000,  # More warmup for stability
            warmup_ratio=0.03,  # Alternative warmup specification
            
            # REGULARIZATION - Prevent overfitting and forgetting
            weight_decay=0.001,  # Less aggressive than your 0.01
            max_grad_norm=2.0,  # Less restrictive than your 1.0
            
            # TRAINING STEPS - Optimized for fast convergence
            max_steps=50000,  # Reduced from 120k for faster training
            eval_steps=1000,  # More frequent evaluation (was 500)
            save_steps=1000,  # More frequent saves (was 500)
            logging_steps=100,  # More frequent logging (was 100)
            
            # MEMORY OPTIMIZATION for RTX 6000 Ada
            gradient_accumulation_steps=accumulation_steps,
            gradient_checkpointing=True,
            dataloader_num_workers=8,  # Increased from 8 for faster data loading
            dataloader_persistent_workers=True,
            dataloader_pin_memory=True,  # Faster CPU->GPU transfer
            
            # MIXED PRECISION - Crucial for speed on Ada architecture
            bf16=True,  # BF16 is optimal for Ada generation
            bf16_full_eval=True,  # Use BF16 for evaluation too
            
            # OPTIMIZER - Better for VLM fine-tuning
            optim="adamw_torch_fused",  # Faster than adafactor on Ada
            adam_beta1=0.9,
            adam_beta2=0.95,  # Better for transformer training
            adam_epsilon=1e-8,
            
            # CHECKPOINTING STRATEGY
            save_strategy='steps',
            save_total_limit=5,  # Keep more checkpoints for safety
            load_best_model_at_end=False, # was true
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            
            # EVALUATION STRATEGY
            eval_strategy='steps',
            eval_accumulation_steps=1,  # Reduce memory during eval
            
            # MISCELLANEOUS
            remove_unused_columns=False,
            report_to=None,
            logging_dir=log_path,
            
            # SPEED OPTIMIZATIONS
            torch_compile=False,  # Set to True if you want to experiment (can be unstable)
            include_inputs_for_metrics=False,  # Reduce memory during evaluation
            ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate, processor=processor, n_points=256),
        tokenizer=processor,
        callbacks=[
            PrintToFileCallback(),
            ClearCUDACacheCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=5,  # Stop after 5 evals without improvement
                early_stopping_threshold=0.001  # Minimum improvement to consider as progress
        )
        ]
    )
    trainer.train()
    trainer.save_model(log_path)
    processor.save_pretrained(log_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--log-path', type=str, default='./work_dirs')
    parser.add_argument('--mode', type=str, default='pc_img')
    parser.add_argument('--use-text', action='store_true')

    args = parser.parse_args()
    run(args.data_path, args.log_path, args.mode, args.use_text)