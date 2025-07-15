import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
torch.cuda.empty_cache()