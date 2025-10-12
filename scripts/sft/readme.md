"""
Revised Supervised Fine-tuning Script for Pi-Zero Model

TODO:
1. when saving model, remove the key-value pair 'type' and replace the cuda:x as cuda
2. Distributed training with DDP is not stable yet
iv) More efficient data loading for multiple GPUs. 
   i) [seems that we fixed it by evaluating with model.module instead of model]saving model on the central not still causes jam...
   ii) [seems that we fixed it with disabling hydra autosave from .yaml] also the .hydra and train_sft.log are still saved by each process to different dirs. maybe because the output.dir is 
specific to the time this hydra in this process is created. output: dir: /nvme_data/tonghe/openpi/results/sft_pi0_maniskill/${now:%Y-%m-%d}_${now:%H-%M-%S}
   maybe the only ddp that works with hydra is the ddp spawn. 
   @turian ddp_spawn is not better but it's the only ddp mode that works correctly with hydra right now.
   As I mentioned, normal DDP generates multiple unwanted files. This is due to the fact that ddp launches a new process for each GPU, which doesn't go well with the way hydra creates different output dir each time a program is launched.
   https://github.com/ashleve/lightning-hydra-template/issues/393
   iii) [seems temporarily not very important] we should use spawn instead. this implementation is not professional. 
   "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
   

New features:
1. EMA (Exponential Moving Average) model support
2. Gradient scaler for mixed precision training
3. Options to evaluate both regular and EMA models
4. Improved checkpoint saving/loading with EMA
5. Size issues:
   1. The SFT model will be saved in torch.bfloat16 and there is a name tying mechanism in saving the language tokenizer,
   which replaces `model.paligemma_with_expert.paligemma.language_model.lm_head.weight` 
   into `model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight:`. 
   2. Besides, it also saves the normalization statistics. 
   3. This makes the saved checkpoint only takes up 7GB instead of 13 GB as of the original pre-trained model. 

Key Features:
- Distributed training on multiple GPUs with DDP
- EMA model for better training stability
- Mixed precision training with gradient scaler
- Checkpoint saving and resuming for both main and EMA models
- Evaluation during training for both models
- Hydra configuration management
- Resume model, ema_model, optimizers and schedulers from checkpoint: 
    specify cfg.training.resume_from_checkpoint as the root ckpt directory that looks like this:
    cfg.training.resume_from_checkpoint
                            ├── model/
                            │   ├── config.json  # policy config
                            │   ├── model.safetensors  # policy weights
                            ├── ema_model/ #(optional)
                            │   ├── config.json  # policy config
                            │   ├── model.safetensors  # policy weights
                            |__ training_state/
                                ├── optimizer_param_groups.json  #  optimizer param groups
                                |── optimizer_state.safetensors  # optimizer state
                                ├── rng_state.safetensors  # rng states
                                ├── scheduler_state.json  # scheduler state
                                └── training_step.json  # training step
    and then this script will automatically load the models from that root dir. 
Usage:
    Single GPU: python train_sft_revised.py
    Multi-GPU: torchrun --standalone --nnodes 1 --nproc-per-node 4 train_sft_revised.py
    
Known issues:
* amp:
    PaliGemma's attention mechanism uses large negative values for masking
    FP16 range is limited: -65,504 to +65,504
    Attention masks typically use -1e9 or -torch.inf → fp16 overflow
    This is a known issue with transformer models + mixed precision
"""