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


#!/usr/bin/env python3

import os
# Disable tokenizer parallelism to avoid deadlocks in multi-GPU training
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
# MEMORY OPTIMIZATION: Enable PyTorch memory optimizations. This one significantly saved GPU DRAM. Previously if your model is 13GB then it needs 13GB for each gpu to load it. but after this, it only needs couple of GB. 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# MEMORY OPTIMIZATION: Reduce memory fragmentation
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# NCCL timeout setting.
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '3600000'  # 1 hour (in miliseconds)
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '3600'  # 
os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '1'
os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '1000'
# Enable debugging info
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import signal
import atexit
import logging
from tqdm import tqdm
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.utils.train_utils import load_training_state
from utils.custom_dirs import PI_R_ROOT_DIR  
from utils.clear_pycache import clean_pycache

from scripts.evaluate.eval_helpers import *
from utils.custom_memory_manager import cleanup_cuda_memory, signal_handler
from utils.custom_distributed import setup_distributed, cleanup_and_killprocess
from utils.ema import update_ema_parameters
from utils.custom_logging import setup_wandb
from sft_helpers import *
import pickle
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.eval_pi0_maniskill import EvalAgent
from scripts.env.domain_rand import DomainRandomization
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register cleanup functions  
atexit.register(cleanup_cuda_memory)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    """Main training function"""
    try:
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed()
        is_main_process = rank == 0
        
        # Ensure all processes wait for gpu remapping.
        if world_size > 1:
            dist.barrier()
    
        # Fix output directory synchronization for distributed training
        if world_size > 1:
            output_dir = Path(cfg.output.dir)
        else:
            # Single process - use original directory (with timestamp from Hydra)
            output_dir = Path(cfg.output.dir)
        
        if is_main_process:
            clean_pycache(PI_R_ROOT_DIR)
            cleanup_cuda_memory()
            logger.info("Starting Pi0 Supervised Fine-tuning with EMA")
            logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
            logger.info(f"Output directory: {output_dir}")
        
        # Setup device
        if cfg.device.model_device == "auto":
            model_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            model_device = torch.device(cfg.device.model_device)
        logger.info(f"model_device={model_device}")
        
        # Evaluation settings
        use_ema = cfg.training.get('use_ema', False)
        eval_main_model = cfg.eval.get('eval_main_model', True)             # test the main model during validation and simulation testing. 
        eval_ema_model = cfg.eval.get('eval_ema_model', True) and use_ema   # to test the EMA model, instead of the original model, during validation and simulation testing. 
        
        # Setup test agent in ManiSkill3 simulator (only on master process)
        test_agent = None
        if is_main_process and cfg.eval.test_in_sim:
            try:
                # Load evaluation config with device overrides
                config_path = Path(__file__).parent.parent / cfg.eval.sim_cfg_path
                cfg_eval = OmegaConf.load(config_path)
                # For simplicity we put the simulator and model on the same device but this can be customized in later versions. 
                cfg_eval.model.device = str(model_device)
                cfg_eval.sim.device = cfg.device.sim_device
                # Overload output settings
                cfg_eval.output.dir=os.path.join(output_dir, 'test')
                cfg_eval.output.save_data=False
                # Apply sim config overrides from main cfg
                if cfg.eval.get('sim_cfg_overrides'):
                    for key, value in cfg.eval.sim_cfg_overrides.items():
                        OmegaConf.update(cfg_eval, key, value, force_add=True)
                        logger.info(f"Applied sim override: {key} = {value}")
                # Import and create evaluation agent
                test_agent: EvalAgent = EvalAgent(cfg_eval)
                logging.info(f"Evaluation agent created successfully on device: {model_device}")
            except Exception as e:
                logging.warning(f"Could not create evaluation agent: {e}")
                raise e

        # Create output directories (only on main process)
        if is_main_process:
            logger.info(f"Creating output directories.")
            output_dir: Path
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub_dir in [output_dir/'last'/'model', output_dir/'best'/'model']:
                os.makedirs(sub_dir, exist_ok=True)
            # Save SFT configuration
            with open(output_dir / "cfg_sft.yaml", "w") as result_f:
                OmegaConf.save(cfg, result_f)
        # Ensure all processes wait for directory creation
        if world_size > 1:
            dist.barrier()
        
        # Setup logging to wandb
        verbose = cfg.training.get('verbose', False)
        verbose_val = cfg.training.get('verbose_val', False)
        if is_main_process and cfg.get('wandb', None) is not None:
            setup_wandb(cfg, logger)
        # Ensure all processes wait for wandb setup
        if world_size > 1:
            dist.barrier()
        
        # Load dataset statistics
        if is_main_process:
            logger.info(f"Loading dataset statistics from {cfg.dataset.normalization_path}")
        try:
            dataset_stats = torch.load(cfg.dataset.normalization_path, map_location='cpu')
            # Create model and ema model (optionally), and wrap them with DDP for data-parallel distributed training. 
            model, ema_model, model_to_eval, ema_model_to_eval = create_sft_model_ema(is_main_process=is_main_process, 
                                                                                        world_size=world_size,
                                                                                        local_rank=local_rank,
                                                                                        cfg=cfg,
                                                                                        model_device=model_device, 
                                                                                        dataset_stats=dataset_stats, 
                                                                                        output_dir=output_dir,
                                                                                        use_ema=use_ema,
                                                                                        eval_ema_model=eval_ema_model)
        except Exception as e:
            logger.error(f"Failed to load dataset statistics or create model: {e}")
            cleanup_and_killprocess(world_size)
            raise e
        
        # Set model to training mode
        model.train()
        if ema_model:
            ema_model.eval()  # EMA model is always in eval mode
        # Create optimizer and learning rate scheduler
        start_step = 0
        optimizer, scheduler, trainable_params= create_optimizer_and_scheduler(model, cfg)
        # Resume from checkpoint if specified
        if cfg.training.resume_from_checkpoint:
            ckpt_path = Path(cfg.training.resume_from_checkpoint)
            if is_main_process:
                logger.info(f"Resume step, optimizer, and scheduler states from: {ckpt_path}")    
            try:
                start_step, optimizer, scheduler = load_training_state(ckpt_path, optimizer, scheduler) # this function automatically grapss the training_state subdir from the root ckpt dir. 
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                cleanup_and_killprocess(world_size)
                raise e
        # Create gradient scaler for mixed precision training
        use_amp = cfg.training.get('use_amp', False)
        grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        # Log AMP status and potential issues
        if use_amp and is_main_process:
            logger.info("Mixed precision (AMP) enabled - may cause fp16 overflow in attention layers")
            logger.info("If you encounter 'value cannot be converted to type at::Half' errors, disable AMP with training.use_amp=false")
        elif not use_amp and is_main_process:
            logger.info("Mixed precision (AMP) disabled - using full fp32 precision")
        # Load SFT dataset (train and val). 
        try:
            use_multi_step = True #cfg.dataset.get('use_multi_step', False)
            horizon_steps = cfg.dataset.horizon_steps
            cond_steps = cfg.dataset.cond_steps
            img_cond_steps = cfg.dataset.img_cond_steps
            # Ensure dataset configuration matches model inputs
            if model_to_eval.config.n_action_steps != horizon_steps:
                raise ValueError(f"Dataset horizon_steps={horizon_steps} doesn't match model n_action_steps={model_to_eval.config.n_action_steps}")
            if model_to_eval.config.n_obs_steps != cond_steps:
                raise ValueError(f"Dataset cond_steps={cond_steps} doesn't match model n_obs_steps={model_to_eval.config.n_obs_steps}")
            train_loader, val_loader, train_sampler=load_sft_dataset(cfg, world_size, is_main_process, use_multi_step=use_multi_step, horizon_steps=horizon_steps, cond_steps=cond_steps, img_cond_steps=img_cond_steps, distributed_val=True)
        except Exception as e:  
            logger.error(f"Failed to load dataset: {e}")
            cleanup_and_killprocess(world_size)
            raise e
        
        # Test in sim before training to see how the pre-trained model performs under current configuration.
        success_rate=0.0
        best_success_rate=0.0
        if test_agent is not None and is_main_process:
            best_success_rate, success_rate, is_current_best_success_rate = test_model_in_sim(step=start_step, test_agent=test_agent, best_success_rate=best_success_rate, success_rate=success_rate, model_to_eval=model_to_eval, ema_model_to_eval=ema_model_to_eval, eval_ema_model=eval_ema_model)
        if is_main_process and cfg.get('wandb', None) is not None:
            wandb.log({"test_success_rate":success_rate, 'step': 0})       
        is_current_best_success_rate=False
        
        # Optionally apply domain randomization
        domain_randomizer = DomainRandomization(cfg.domain_randomization) if cfg.get("domain_randomization", False) else None
        if is_main_process:
            if domain_randomizer:
                logger.info(f"Domain randomization enabled during training with settings:{OmegaConf.to_yaml(cfg.domain_randomization)}")
            else:
                logger.info("No domain randomization applied")
        dist.barrier()  # Ensure all processes wait for initial sim testing to complete
        if is_main_process:
            logger.info(f"Training starts:")
        # Training loop
        model.train()
        if ema_model:
            ema_model.eval()
            # EMA configuration
            ema_alpha = cfg.training.get('ema_alpha', 0.995)
            ema_update_every = cfg.training.get('ema_update_every', 1)
        
        step = start_step
        recent_losses = []
        # Cache cfg values that will occur in the training loop
        n_epoch = cfg.training.n_epoch
        max_train_step=cfg.training.max_steps
        save_frequency=cfg.training.save_freq  # save checkpoint every this many steps (batch). 
        eval_freq=cfg.eval.eval_steps  # evaluate every `eval_freq` steps (batch). 
        if save_frequency != eval_freq and is_main_process:
            logger.warning(f"save_frequency={save_frequency} != eval_freq={eval_freq}. This may cause the model to be saved at different steps than the evaluation steps, making evaluation results less useful.")
        log_freq=cfg.logging.log_freq  # log every `log_freq` steps (batch). 
        num_eval_batches=cfg.eval.num_eval_batches
        grad_acccum_steps=cfg.training.grad_accumulation_steps
        grad_clip_norm=cfg.training.get('grad_clip_norm', 10.0)
        lr=cfg.training.learning_rate
        warmup_steps=cfg.training.warmup_steps
        
        if is_main_process:
            if use_ema:
                logger.info(f"EMA alpha: {ema_alpha}, update every: {ema_update_every} steps")
            logger.info(f"Evaluation: main_model={eval_main_model}, ema_model={eval_ema_model}")        
        # Initialize gradient norm tracking and lowest eval loss
        current_grad_norm = 0.0
        current_lowest_eval_loss = float('inf')
        results=[{} for _ in range(max_train_step)]
        for epoch in tqdm(range(n_epoch), desc="Training Epoch", dynamic_ncols=True, disable=not is_main_process):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            for batch in tqdm(train_loader, desc="Training Batch", dynamic_ncols=True, disable=not is_main_process):
                if step >= max_train_step:
                    break
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(model_device, non_blocking=True)
                # Apply domain randomization during training when applicable (this is in-place)
                if not (cfg.eval.enabled and step % eval_freq == 0 and step > 0) and (domain_randomizer is not None): 
                    batch = domain_randomizer.apply(batch)
                # Forward pass
                try:
                    loss, loss_dict = compute_loss(model, batch, model_device, use_amp, verbose=verbose)
                except Exception as e:
                    error_msg = str(e)
                    if "value cannot be converted to type at::Half without overflow" in error_msg:
                        logger.error(f"FP16 overflow error at step {step}! This is a known issue with mixed precision training.")
                        logger.error("SOLUTION: Disable mixed precision by adding 'training.use_amp=false' to your command")
                        logger.error("Example: python train_sft.py ... training.use_amp=false")
                    else:
                        logger.error(f"Error in forward pass at step {step}: {e}")
                    cleanup_and_killprocess(world_size)
                    raise e
                # Backward pass with gradient scaling
                try:
                    grad_scaler.scale(loss).backward()
                except Exception as e:
                    logger.error(f"Error in backward pass at step {step}: {e}")
                    cleanup_and_killprocess(world_size)
                    raise e
                # Optimizer step with gradient accumulation
                if (step + 1) %  grad_acccum_steps== 0:
                    if dist.is_initialized() and grad_acccum_steps>1:
                        dist.barrier()  # Sync before optimizer step
                    # Unscale gradients for clipping
                    grad_scaler.unscale_(optimizer)
                    # Gradient clipping and norm calculation. current_grad_norm is the grad norm before clipping. 
                    current_grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, 
                        grad_clip_norm
                    )
                    # Optimizer step
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    # Update learning rate after warmup
                    if step < warmup_steps:
                        # Linear warmup
                        warmup_lr = lr * (step + 1) / warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = warmup_lr
                    else:
                        if scheduler is not None:
                            scheduler.step()
                    # Update EMA model
                    if use_ema and ema_model and step % ema_update_every == 0 :
                        update_ema_parameters(ema_model, model, ema_alpha)
                # Track metrics
                recent_losses.append(loss_dict['l2_loss'])
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                # Logging
                if is_main_process and step % log_freq == 0:
                    train_loss = loss_dict['l2_loss']
                    logger.info(f"Step {step}, Loss: {train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}, Grad Norm: {current_grad_norm:.4f}")
                    if cfg.wandb:
                        train_log_dict = {
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'step': step,
                            'grad_norm_before_clipping': current_grad_norm.item() if isinstance(current_grad_norm, torch.Tensor) else current_grad_norm,
                        }
                        train_log_dict.update(loss_dict)
                        results[step].update(train_log_dict)
                        wandb.log(train_log_dict)
                        
                # Evaluation: distributed validate and synchronized test in sim.
                eval_metrics = {}
                is_lowest_val_loss = False
                # Evaluate every `eval_freq` steps (batch). 
                if cfg.eval.enabled and step % eval_freq == 0 and step > 0 :
                    if is_main_process:
                        logger.info(f"Synchronize for validation...")
                    if world_size > 1:
                        dist.barrier()
                    if is_main_process:
                        logger.info(f"#"*100+f"\nDistributed Validation at step={step}, epoch={epoch}")
                    # Evaluate main model (val_loader is guaranteed to be non-None on main process)
                    if eval_main_model:
                        main_metrics = validate_model_distributed(model_to_eval=model_to_eval, val_loader=val_loader, device=model_device, use_amp=use_amp, num_eval_batches=num_eval_batches, verbose=verbose_val, show_progress=True)
                        if is_main_process:
                            eval_metrics.update({f'main_{k}': v for k, v in main_metrics.items()})
                    # Evaluate EMA model when enabled
                    if use_ema and ema_model and eval_ema_model:
                        ema_metrics = validate_model_distributed(model_to_eval=ema_model_to_eval, val_loader=val_loader, device=model_device, use_amp=use_amp, num_eval_batches=num_eval_batches, verbose=verbose_val, show_progress=True)
                        if is_main_process:
                            eval_metrics.update({f'ema_{k}': v for k, v in ema_metrics.items()})
                    # Synchronize all processes after distributed validation
                    if world_size > 1:
                        dist.barrier()
                        logger.info(f"Rank {rank}: Validation complete")
                    # Determine if this is the best checkpoint (using EMA model if available)
                    if is_main_process:
                        current_eval_loss = eval_metrics.get('ema_val_loss', eval_metrics.get('main_val_loss', float('inf')))
                        is_lowest_val_loss = current_eval_loss < current_lowest_eval_loss
                        if is_lowest_val_loss:
                            current_lowest_eval_loss = current_eval_loss
                        logger.info(f"Current eval loss: {current_eval_loss:.6f}, current lowest eval loss: {current_lowest_eval_loss:.6f}, is_lowest_val_loss? {is_lowest_val_loss}")
                        # Evaluate in sim
                        if test_agent is not None:
                            best_success_rate, success_rate, is_current_best_success_rate= test_model_in_sim(step=step, test_agent=test_agent, best_success_rate=best_success_rate, success_rate=success_rate, model_to_eval=model_to_eval, ema_model_to_eval=ema_model_to_eval, eval_ema_model=eval_ema_model)
                            eval_metrics.update({'test_success_rate': success_rate})
                            
                        else:
                            logger.info(f"Skipping simulation test as eval_agent={test_agent} is None. ")
                            is_current_best_success_rate=False
                        # Report and log eval metrics
                        for key, value in eval_metrics.items():
                            logger.info(f"Step {step}, {key}: {value:.6f}")
                        if cfg.wandb:
                            wandb.log({**eval_metrics, 'step': step})
                        results[step].update(eval_metrics)
                    else:
                        # Non-main processes don't do sim testing and don't track validation metrics, they just do the compute. 
                        is_lowest_val_loss = False
                        is_current_best_success_rate = False
                # Return to training mode
                torch.cuda.empty_cache()
                model.train()
                # if dist.is_initialized():
                #     dist.barrier()
                # if is_main_process:
                #     logger.info(f"Will save logs and models. ")
                # Save logs and models
                if is_main_process:
                    # Save results to file
                    with open(os.path.join(output_dir, "train_results.pkl"), "wb") as result_f:
                        pickle.dump(results, result_f)
                    # Save checkpoint every `save_frequency` steps (batch). 
                    if step % save_frequency == 0 and step > 0:
                        
                        logger.info(f"Saving checkpoint to {output_dir} at step {step}")
                        is_current_best = is_current_best_success_rate # is_lowest_val_loss
                        save_sft_model_ema_best(checkpoint_dir=output_dir, step=step, is_current_best=is_current_best,
                                                model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler)
                        logger.info(f"Checkpoint saved successfully at step {step} to {output_dir}")
                # Synchronize all processes after checkpoint saving
                if world_size > 1:
                    dist.barrier()
                step += 1
                if step >= max_train_step:
                    break
            if step >= max_train_step:
                break
        # Ensure all processes wait for final cleanup
        if world_size > 1:
            dist.barrier()
        # Cleanup
        if is_main_process:
            if cfg.wandb:
                wandb.finish()
            if result_f:
                result_f.close()
        cleanup_and_killprocess(world_size)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        cleanup_and_killprocess(world_size)
        return
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in training: {e}")
        cleanup_and_killprocess(world_size)
        raise e
if __name__ == "__main__":
    main() 