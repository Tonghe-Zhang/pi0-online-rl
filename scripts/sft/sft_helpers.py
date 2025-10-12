# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import copy
from omegaconf import OmegaConf, ListConfig, DictConfig
import json
from pathlib import Path
from tqdm import tqdm as tqdm
from contextlib import nullcontext
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lerobot.common.policies.pretrained import PreTrainedPolicy 
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.utils.train_utils import save_training_state
import dataclasses
from safetensors.torch import save_model as save_model_as_safetensor
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.constants import CONFIG_NAME
import gc
import logging
logger = logging.getLogger(__name__)

def create_model(cfg: DictConfig, dataset_stats: Dict[str, Any], device: torch.device) -> tuple[PI0Policy, PI0Config]:
    """Create and initialize the Pi0 model"""
    model_path = cfg.model.path
    logger.info(f"Loading model from {model_path}")
    # Load original model config
    model_config: PI0Config = PI0Config.from_pretrained(model_path)
    # Overridge model_config with command line inputs in training config file
    config_overrides = cfg.model.get('config_overrides', {})
    config_overrides = {k: v for k, v in config_overrides.items() if v is not None} if config_overrides else None
    if config_overrides:
        logger.info("Applying config overrides:")
        for key, value in config_overrides.items():
            if value is not None:
                original_value = getattr(model_config, key, None)
                logger.info(f"  Overriding {key}: {original_value} -> {value}")
                setattr(model_config, key, value)
    # Set training configuration to model config. 
    model_config.freeze_vision_encoder = cfg.model.freeze_vision_encoder
    model_config.train_expert_only = cfg.model.train_expert_only
    model_config.device = str(device)
    
    # Create model from the overridden model config. 
    model = PI0Policy.from_pretrained(
        pretrained_name_or_path=model_path,
        config=model_config,
        dataset_stats=dataset_stats
    )
    model.to(device)
    logger.info(f"Successfully loaded Pi-zero model on {device}")
    
    return model, model_config

def create_ema_model(model: PI0Policy, device: torch.device) -> PI0Policy:
    """Create EMA model as a deep copy of the main model"""
    logger.info("Creating EMA model")
    ema_model = copy.deepcopy(model)
    ema_model.to(device)
    ema_model.eval()  # EMA model is always in eval mode
    
    # Disable gradients for EMA model
    for param in ema_model.parameters():
        param.requires_grad = False
    
    return ema_model

def create_optimizer_and_scheduler(model: nn.Module, cfg: DictConfig):
    """Create optimizer and learning rate scheduler, which only applies to the trainable parameters of the model. 
    Default optimizer is AdamW with cosine annealing learning rate scheduler. 
    Before the warmup steps, the learning rate is linearly increased from 0 to the default learning rate. 
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")
    
    # Create optimizer
    optimizer = AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.optimizer.betas),  # There is a ListConfig in the optimizer config, which is not JSON serializable. # So we need to convert it to a regular Python object before passing it to the optimizer. 
        eps=cfg.optimizer.eps,
        weight_decay=cfg.training.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_steps - cfg.training.warmup_steps,
        eta_min=cfg.scheduler.min_lr
    )
    
    return optimizer, scheduler, trainable_params

def save_model_config(model_config, output_dir, logger):
    # Convert output_dir to Path object to support / operator
    output_dir = Path(output_dir)
    
    try:
        # Method 1: Convert dataclass to dict (primary method)
        import dataclasses
        config_dict = dataclasses.asdict(model_config)
        with open(output_dir / "model_config.yaml", "w") as f:
            OmegaConf.save(OmegaConf.create(config_dict), f)
        logger.info(f"Model config saved as YAML to {output_dir / 'model_config.yaml'}")
    except Exception as e:
        logger.warning(f"Failed to save model config as YAML: {e}")
        try:
            # Fallback: Save as JSON
            import json
            import dataclasses
            config_dict = dataclasses.asdict(model_config)
            with open(output_dir / "model_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Model config saved as JSON to {output_dir / 'model_config.json'}")
        except Exception as e2:
            logger.error(f"Failed to save model config in any format: {e2}")
                    
def compute_loss(model: PI0Policy, batch: Dict[str, Any], device: torch.device, use_amp: bool = False, verbose: bool = False) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the flow matching loss for pi0 with mixed precision support"""
    # Handle both single-step and multi-step data formats
    rgb_image = batch['observation.images.top'].to(device)
    proprioception = batch['observation.state'].to(device)
    instructions = batch['task']
    action = batch['action'].to(device)
    
    # Check if this is multi-step data (has extra time dimension)
    if proprioception.dim() == 3:  # [B, cond_steps, state_dim] -> multi-step
        # For multi-step, we need to handle the temporal dimensions
        # Take the most recent observation for conditioning
        proprioception = proprioception[:, -1, :]  # [B, state_dim]
        
        # Handle image conditioning - take most recent image  
        if rgb_image.dim() == 5:  # [B, img_cond_steps, 3, H, W]
            rgb_image = rgb_image[:, -1, :, :, :]  # [B, 3, H, W]
        
        # Action might be multi-step: [B, horizon_steps, action_dim]
        if verbose:
            logger.debug(f"Multi-step format detected:")
            logger.debug(f"  rgb_image shape: {rgb_image.shape}")
            logger.debug(f"  proprioception shape: {proprioception.shape}")
            logger.debug(f"  action shape: {action.shape}")
        
    else:  # Single-step format: [B, state_dim], [B, 3, H, W], [B, action_dim]
        if verbose:
            logger.debug(f"Single-step format detected:")
            logger.debug(f"  rgb_image shape: {rgb_image.shape}")
            logger.debug(f"  proprioception shape: {proprioception.shape}")
            logger.debug(f"  action shape: {action.shape}")
    
    # Prepare batch for model
    batch_on_device = {
        "observation.images.top": rgb_image,  # Tensor[B, C, H, W], this is transposed from simulator output [B,H,W,C], because PaliGemma image encoder receives images like [B,C,H,W], see code PATH_TO_YOUR_CONDA/envs/pi_r/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py function `get_image_features`
        "observation.state": proprioception,  # Tensor[B, state_dim]
        "task": instructions,                 # List of string of length B
        "action": action,                 # List of string of length B
    }
    
    # Forward pass with mixed precision - match train.py pattern
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        if verbose:
            print(f"batch_on_device={batch_on_device.keys()}")
            print(f"batch_on_device['task']={batch_on_device['task']}")
            print(f"batch_on_device['action']={batch_on_device['action'].shape}")
            print(f"batch_on_device['observation.images.top']={batch_on_device['observation.images.top'].shape}")
            print(f"batch_on_device['observation.state']={batch_on_device['observation.state'].shape}")
        loss, loss_dict = model.forward(batch_on_device, verbose=verbose)
    
    return loss, loss_dict

@torch.no_grad
def validate_model(model_to_eval: PI0Policy| DDP, val_loader: DataLoader, device: torch.device, use_amp: bool = False, num_eval_batches=None, verbose: bool = False) -> Dict[str, float]:
    """
    1. Evaluate validation loss. 
    Evaluate the model's loss on the validation set for the first num_eval_batches. 
    If num_eval_batches is None, evaluate on the entire validation set. 
    2. Test success rate. 
    Test the model's success rate in the simulation environment that generates the pre-training data. 
    """
    # evaluate only the .module if you use distributed data parallel to avoid hang-ups when using dist.barrier()
    # refs: https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522
    
    model_to_eval.eval()
    total_loss = 0.0
    num_batches = 0
    torch.cuda.empty_cache()
    total_batches = min(len(val_loader), num_eval_batches) if num_eval_batches is not None else len(val_loader)
    with torch.no_grad():
        # for batch_idx, batch in tqdm(enumerate(val_loader), total=total_batches, desc="Validating loss"):
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx % (total_batches // 5 + 1) == 0:
                logger.info(f"Validating batch {batch_idx+1}/{total_batches}...")
            if num_eval_batches is not None and batch_idx >= num_eval_batches:
                break
            loss, loss_dict = compute_loss(model_to_eval, batch, device, use_amp, verbose=verbose)
            total_loss += loss.item()   # l2 loss. 
            num_batches += 1
    torch.cuda.empty_cache()
    
    # the average loss on the eval batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation loss: avg_loss={avg_loss}")
    return {'val_loss': avg_loss}


def convert_omegaconf_to_python(obj):
    """Recursively convert OmegaConf objects to regular Python objects"""
    if isinstance(obj, (ListConfig, DictConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, (list, tuple)):
        return [convert_omegaconf_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_omegaconf_to_python(v) for k, v in obj.items()}
    else:
        return obj


def preprocess_optimizer_for_saving(optimizer: Optimizer) -> Optimizer:
    """
    Create a copy of the optimizer with OmegaConf objects converted to regular Python objects.
    This prevents JSON serialization errors when saving the optimizer state.
    """
    import copy
    
    # Create a shallow copy of the optimizer
    optimizer_copy = copy.copy(optimizer)
    
    # Deep copy the param_groups to avoid modifying the original
    optimizer_copy.param_groups = copy.deepcopy(optimizer.param_groups)
    
    # Convert OmegaConf objects in param_groups to regular Python objects
    for param_group in optimizer_copy.param_groups:
        for key, value in param_group.items():
            if isinstance(value, (ListConfig, DictConfig)):
                param_group[key] = convert_omegaconf_to_python(value)
    
    return optimizer_copy


def save_sft_model_ema_best(checkpoint_dir: Path,
                        step: int,
                        is_current_best: bool, 
                        model:PreTrainedPolicy,
                        ema_model: Optional[PreTrainedPolicy],
                        optimizer: Optimizer,
                        scheduler: LRScheduler | None = None):
    """This function creates the following directory structure:
    
    checkpoint_dir/
    |__ last
    |    ├── model/
    |    │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
    |    │   ├── model.safetensors  # policy weights
    |    ├── ema_model/ #(optional)
    |    │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
    |    │   ├── model.safetensors  # policy weights
    |    |__ training_state/
    |        ├── optimizer_param_groups.json  #  optimizer param groups
    |        |── optimizer_state.safetensors  # optimizer state
    |        ├── rng_state.safetensors  # rng states
    |        ├── scheduler_state.json  # scheduler state
    |        └── training_step.json  # training step
    |__ best
        ├── model/
        │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
        │   ├── model.safetensors  # policy weights
        ├── ema_model/ #(optional)
        │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
        │   ├── model.safetensors  # policy weights
        |__ training_state/
            ├── optimizer_param_groups.json  #  optimizer param groups
            |── optimizer_state.safetensors  # optimizer state
            ├── rng_state.safetensors  # rng states
            ├── scheduler_state.json  # scheduler state
            └── training_step.json  # training step

    Args:
        checkpoint_dir (Path): The root directory to save the checkpoints
        step (int): The training step at that checkpoint.
        is_current_best (bool): 
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        
    """
    # save the last checkpoint
    save_training_state(checkpoint_dir/'last', step, optimizer, scheduler)
    model_to_save=model.module if isinstance(model, DDP) else model
    
    # Use our custom save function that fixes the 'type' and device issues
    save_model_core(model_to_save, save_directory=checkpoint_dir/'last'/'model')
    if ema_model is not None:
        ema_model_to_save=ema_model.module if isinstance(ema_model, DDP) else ema_model
        save_model_core(ema_model_to_save, save_directory=checkpoint_dir/'last'/'ema_model')
    
    # save the best checkpoint
    if is_current_best:
        save_training_state(checkpoint_dir/'best', step, optimizer, scheduler)
        save_model_core(model_to_save, save_directory=checkpoint_dir/'best'/'model')
        if ema_model is not None:
            save_model_core(ema_model_to_save, save_directory=checkpoint_dir/'best'/'ema_model')

def save_config_core(config: PI0Config, save_directory: Path) -> None:
    """Custom save function that fixes the 'type' field and device issues"""
    
    # Convert config to dict using dataclasses
    
    config_dict = dataclasses.asdict(config)
    
    # Remove the 'type' field if it exists
    if 'type' in config_dict:
        del config_dict['type']
    
    # Normalize device field - convert "cuda:0" to "cuda"
    if 'device' in config_dict and config_dict['device'] is not None:
        device_str = str(config_dict['device'])
        if device_str.startswith('cuda:'):
            config_dict['device'] = 'cuda'
    
    # Save the cleaned config
    with open(save_directory / CONFIG_NAME, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    logger.info(f"Saved config to {save_directory / CONFIG_NAME} with fixes (removed 'type' field and normalized device)")

def save_model_core(model: PreTrainedPolicy, save_directory: Path) -> None:
    """Custom save function that uses the fixed config save method"""
    # Save config with fixes (only for PI0Policy)
    if isinstance(model, PI0Policy):
        save_config_core(model.config, save_directory)
    else:
        # Fall back to default config save for other policy types
        model.config._save_pretrained(save_directory)
    
    # Save model weights normally
    model_to_save = model.module if hasattr(model, "module") else model
    save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
    
    logger.info(f"Saved model to {save_directory} with config fixes")


# Load SFT dataset
from torch.utils.data.distributed import DistributedSampler
from scripts.sft.sft_dataset import PiZeroSFTDataset, PiZeroSFTDatasetMultiStep, PiZeroSFTDatasetMultiStepIterable
def load_sft_dataset(cfg: DictConfig, world_size:int, is_main_process: bool, use_multi_step:bool, horizon_steps:int, cond_steps:int, img_cond_steps:int, distributed_val:bool=False) -> tuple[DataLoader, DataLoader, int, int, int]:
    logger.info(f"Loading SFT dataset with use_multi_step={use_multi_step}, horizon_steps={horizon_steps}, cond_steps={cond_steps}, img_cond_steps={img_cond_steps}")
    if cfg.dataset.use_iterable:
        # When the dataset is super large, e.g. over 1TB, we shard the datasets into multiple files and load from them with a iterable dataset loader. 
        if use_multi_step:
            if distributed_val or (not distributed_val and is_main_process):
                val_dataset = PiZeroSFTDatasetMultiStepIterable(cfg.dataset.shard_metadata_path_val, horizon_steps=horizon_steps, cond_steps=cond_steps, img_cond_steps=img_cond_steps, num_workers=0)
                logger.info(f"Validation dataset created.")
            train_dataset = PiZeroSFTDatasetMultiStepIterable(cfg.dataset.shard_metadata_path_train, horizon_steps=horizon_steps, cond_steps=cond_steps, img_cond_steps=img_cond_steps, num_workers=0) # not supporting multi-processing yet.
            logger.info(f"Train dataset created.")
        else:
            raise NotImplementedError("Single-step iterable dataset is not implemented yet")
    else:
        # When the dataset is not super large (less than the VRAM of GPUs), we directly load the dataset to map location from a single file. 
        dataset_path = cfg.dataset.path
        dataset_map_location = cfg.dataset.map_location
        if is_main_process:
            logger.info(f"Loading dataset from {dataset_path} to {dataset_map_location}")    
        data = torch.load(dataset_path, map_location=dataset_map_location, mmap=True)
        # Create Dataset objects
        if use_multi_step:
            if is_main_process:
                logger.info(f"Using multi-step dataset with horizon_steps={horizon_steps}, cond_steps={cond_steps}, img_cond_steps={img_cond_steps}")
                val_dataset = PiZeroSFTDatasetMultiStep(
                data['val'],
                horizon_steps=horizon_steps,
                cond_steps=cond_steps,
                img_cond_steps=img_cond_steps
            )
            logger.info(f"Validation dataset created.")
            train_dataset = PiZeroSFTDatasetMultiStep(
                data['train'],
                horizon_steps=horizon_steps,
                cond_steps=cond_steps,
                img_cond_steps=img_cond_steps
            )
            logger.info(f"Train dataset created.")
        else:
            if is_main_process:
                logger.info("Using single-step dataset")
                val_dataset = PiZeroSFTDataset(data['val'])
                logger.info(f"Validation dataset created.")
            train_dataset = PiZeroSFTDataset(data['train'])
            logger.info(f"Train dataset created.")
    
    # Create distributed data loaders
    logger.info(f"Creating dataset loaders...")
    train_sampler = DistributedSampler(train_dataset, shuffle=cfg.dataset.shuffle) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(cfg.dataset.shuffle and train_sampler is None),
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    logger.info(f"Training loader created with {len(train_loader)} batches, batch_size={train_loader.batch_size}")
    if distributed_val: 
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)# Don't shuffle validation data, # Keep all validation samples
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.training.num_val_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Distributed validation loader created with {len(train_loader)} batches, batch_size={train_loader.batch_size}")
    else:
        # Create non-distributed validation loader for consistent evaluation (only on main process)
        val_loader = None
        if is_main_process:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_val_workers,
                pin_memory=True
            )
            logger.info(f"Validation loader created with {len(val_loader)} batches, batch_size={val_loader.batch_size}")

    return train_loader, val_loader, train_sampler


def create_sft_model_ema(is_main_process:bool, world_size:int, local_rank:int, cfg, model_device: torch.device, dataset_stats, output_dir, use_ema: bool, eval_ema_model:bool) -> tuple[PI0Policy, Optional[PI0Policy], DDP|PI0Policy, Optional[DDP|PI0Policy]]:
    """
    Create the model and EMA model, and wrap them with DDP for data-parallel distributed training. 
    Unwrap them for evaluation later. 
    Args:
        is_main_process (bool): Whether this is the main process.
        world_size (int): The number of processes in the distributed training.
        local_rank (int): The rank of the current process.
        cfg (DictConfig): The configuration.
        model_device (torch.device): The device to load the model on.
    """
    
    
    # Load model on device
    if is_main_process:
        logger.info(f"Loading Pi-zero model to {model_device}")
    # Optionally overload model checkpoint with resume path.
    if cfg.training.resume_from_checkpoint:
        ckpt_path = cfg.training.resume_from_checkpoint
        cfg.model.path = os.path.join(ckpt_path, 'model') # this dir should contain a file named model.safetensors and config.json
        if is_main_process:
            logger.info(f"Resume model from: {cfg.model.path}")   
    # Load model on device
    model_and_config=create_model(cfg, dataset_stats, model_device)
    model: PI0Policy=model_and_config[0]
    model_config: PI0Config=model_and_config[1]
    # Save model config using proper serialization
    if is_main_process:
        save_model_config(model_config, output_dir, logger)
    # Create EMA model conditionally
    ema_model: Optional[PI0Policy] = None
    if use_ema:
        # Optionally overload ema_model checkpoint with resume path.
        if cfg.training.resume_from_checkpoint:
            ckpt_path = cfg.training.resume_from_checkpoint
            cfg.model.path = os.path.join(ckpt_path, 'ema_model') # this dir should contain a file named model.safetensors and config.json
            if is_main_process:
                logger.info(f"Resume ema_model from: {cfg.model.path}")   
        else:
            ema_model = create_ema_model(model, model_device)
            if is_main_process:
                logger.info(f"Created ema_model from scratch. ")   
        if is_main_process:
            logger.info("EMA model enabled")
        if is_main_process:
            for sub_dir in [output_dir/'last'/'ema_model', output_dir/'best'/'ema_model']:
                os.makedirs(sub_dir, exist_ok=True)
    else:
        if is_main_process:
            logger.info("EMA model disabled")
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {n_trainable_params:,}")
        logger.info(f"Trainable ratio: {100.0 * n_trainable_params / total_params:.2f}%")
    
    # Wrap models with DDP for distributed training. Unwrapp them for evaluation later. 
    if world_size > 1:
        logger.info(f"Wrapping models with DDP to local_rank={local_rank}")    
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)  # type: ignore
        if ema_model:
            ema_model = DDP(ema_model, device_ids=[local_rank], find_unused_parameters=True)  # type: ignore
    model_to_eval=model.module if isinstance(model, DDP) else model
    if use_ema and ema_model and eval_ema_model:
        ema_model_to_eval=ema_model.module if isinstance(ema_model, DDP) else ema_model
    else:
        ema_model_to_eval=None

    return model, ema_model, model_to_eval, ema_model_to_eval


from utils.custom_memory_manager import cleanup_cuda_memory
from scripts.evaluate.eval_pi0_maniskill import EvalAgent
from typing import Tuple
@torch.no_grad
def test_model_in_sim(step:int, test_agent: Optional[EvalAgent], best_success_rate:float, success_rate:float,model_to_eval: PI0Policy, ema_model_to_eval: Optional[PI0Policy], eval_ema_model:bool)->Tuple[float, float, bool]:
    cleanup_cuda_memory()
    is_current_best_success_rate = False
    if test_agent.save_videos:
        test_agent.video_dir=Path(test_agent.output_dir) / 'videos'/ f'step_{step}'
        os.makedirs(test_agent.video_dir, exist_ok=True)       
        logger.info(f"Preparing to save videos to {test_agent.video_dir}")
    if ema_model_to_eval and eval_ema_model:
        success_rate=test_agent.test(ema_model_to_eval, verbose=False)
    else:
        success_rate=test_agent.test(model_to_eval, verbose=False)
    logger.info(f"Tested in sim before training: success_rate={success_rate:.2f}")
    if success_rate > best_success_rate:
        logger.info(f"New best success rate: {success_rate:.2f} (previous: {best_success_rate:.2f})")
        best_success_rate = success_rate
        is_current_best_success_rate = True
    return best_success_rate, success_rate, is_current_best_success_rate



import torch.distributed as dist
@torch.no_grad
def validate_model_distributed(model_to_eval: PI0Policy | DDP, val_loader: DataLoader, device: torch.device, 
                              use_amp: bool = False, num_eval_batches: Optional[int] = None, 
                              verbose: bool = False, show_progress: bool = True) -> Dict[str, float]:
    """
    Distributed validation with internal synchronization - NO external barriers needed.
    """
    # Get distributed info
    if dist.is_initialized():
        rank = dist.get_rank()
        is_distributed = True
    else:
        rank = 0
        is_distributed = False
    
    is_main_process = rank == 0
    
    model_to_eval.eval()
    total_loss = 0.0
    num_batches = 0
    torch.cuda.empty_cache()
    # progress_bar = tqdm(enumerate(val_loader), total=total_batches, desc="Validating loss", disable=not (show_progress and is_main_process))
    
    with torch.no_grad():
        total_batches = num_eval_batches if num_eval_batches is not None else len(val_loader)        # total_batches = num_eval_batches if num_eval_batches is not None else len(progress_bar)
        report_intervals = [int(total_batches * i / 5) for i in range(1, 6)]  # 20%, 40%, ..., 100%

        for batch_idx, batch in enumerate(val_loader):         # for batch_idx, batch in progress_bar:
            if num_eval_batches is not None and batch_idx >= num_eval_batches:
                break
            
            loss, loss_dict = compute_loss(model_to_eval, batch, device, use_amp, verbose=verbose and is_main_process)
            total_loss += loss.item()
            num_batches += 1

            if show_progress and is_main_process and batch_idx+1 in report_intervals:
                # progress_bar.set_postfix({
                #     'progress': f'{batch_idx} / {total_batches}',
                #     'loss': f'{loss.item():.4f}'
                # })
                progress_percent = ((batch_idx + 1) / total_batches) * 100
                logger.info(f"Validation Progress: {progress_percent:.0f}% ({batch_idx + 1}/{total_batches})")

    torch.cuda.empty_cache()
    gc.collect()
    
    # SYNCHRONIZATION: Aggregate results across all processes
    if is_distributed:
        # Create tensors for reduction
        metrics_tensor = torch.tensor([total_loss, float(num_batches)], device=device, dtype=torch.float64)
        # All-reduce to sum across all processes (this is the key synchronization point)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        # Extract aggregated values (now all processes have the same values)
        global_total_loss = metrics_tensor[0].item()
        global_num_batches = int(metrics_tensor[1].item())
    else:
        global_total_loss = total_loss
        global_num_batches = num_batches
    
    # Calculate average loss (same result on all processes)
    avg_loss = global_total_loss / global_num_batches if global_num_batches > 0 else 0.0
    
    if is_main_process:
        logger.info(f"Validation loss: avg_loss={avg_loss:.6f} (batches={global_num_batches})")
    # NO BARRIER HERE - all_reduce already synchronized all processes
    return {'val_loss': avg_loss}