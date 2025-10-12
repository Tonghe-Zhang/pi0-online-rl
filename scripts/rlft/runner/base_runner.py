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
# Logging and visualize
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
# Custom libraries
from utils.reproduce import set_seed_everywhere
from utils.custom_logging import setup_wandb
from scripts.evaluate.eval_helpers import save_config
# Blue text formatting for success messages
from termcolor import colored
class BaseRunner:
    def __init__(self, cfg):
        ###############################################################################
        # Caching omega config
        self.cfg = cfg
        # Sim env
        self.env_id=cfg.env.id
        self.n_envs=cfg.env.num_envs
        self.seed=cfg.seed
        self.same_seed_for_all_envs = cfg.get("same_seed_for_all_envs", False)   # when toggled, use the same seed for all envs. else, use seed+i for the i-th env. 
        self.sim_config=cfg.env.sim_config
        self.sensor_config=cfg.env.sensor_configs
        self.control_mode=cfg.env.control_mode
        self.obs_mode=cfg.env.obs_mode
        self.reset_at_iteration=cfg.env.reset_at_iteration
        # Domain randomization
        self.apply_domain_randomization = self.cfg.get("domain_randomization", False)
        if self.apply_domain_randomization:
            self.domain_rand_cfg= self.cfg.domain_randomization
        # Episode and rollout
        self.n_steps_episode=cfg.env.max_episode_len                    # this one is passed to the environment constructor to define maximum interaction steps per episode. When elapsed_steps is larger than this value, environment returns `truncated`. 
        self.n_steps_rollout=cfg.env.n_steps_rollout                    # number of environment interactions per rollout in each iteration. this can be greater than the n_steps_episode, since we do autoreset between episodes. 
        self.reset_options_venv = cfg.env.get("reset_options", {})      # do NOT define env_idx here because it interrupts with domain randomization. 
        # Training iterations        
        self.itr = 0
        self.n_train_itr=cfg.train.n_train_itr                             # PPO iterations, how many rollouts in the whole training procedure. 
        self.total_steps = self.n_steps_rollout * self.n_envs           # total number of rounds of model inference. The total number of samples (env interactions) is  self.total_steps x self.act_steps
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr        # Warm up period for critic before actor updates        
        self.save_model_freq = cfg.train.save_model_freq
        self.skip_initial_eval = cfg.train.skip_initial_eval
        self.log_freq = cfg.train.log_freq
        self.val_freq = cfg.train.val_freq
        self.record_video_condition = cfg.train.record_video_condition
        self.video_freq = cfg.train.video_freq
        # Resume
        self.resume_dir = cfg.get('resume_dir', None)
        self.resume = self.resume_dir is not None
        # IO paths
        self.normalization_path=cfg.dataset.normalization_path
        self.model_path=cfg.model.path
        self.output_dir=os.path.join(cfg.output.dir)
        # Logging, rendering, and data/checkpoints saving
        self.log_dir = os.path.join(self.output_dir, "log")              # configuration files, architecture, checkpoints. 
        self.result_path = os.path.join(self.log_dir, "result.pkl")
        self.save_trajs = cfg.train.get("save_trajs", False)
        self.render_dir = os.path.join(self.output_dir, "render")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        self.verbose_update = cfg.logging.verbose_update
        self.verbose_input = cfg.logging.verbose_input # show input batch to the model. 
        self.save_data=cfg.output.save_data
        self.save_videos=cfg.output.save_videos
        # Set global random seed
        set_seed_everywhere(seed=self.seed)
        # Wandb
        self.use_wandb = cfg.get('wandb', None) is not None
        if self.use_wandb:
            setup_wandb(self.cfg, logger)
        # Setup device assignment, environment, domain randomization, model, optimizer, learning rate scheduler for training. 
        self.setup_pipeline()
        
    def run(self):
        pass
    
    def set_devices(self):
        pass
    
    def create_env(self):
        pass
    
    def build_domain_randomization(self):
        pass
    
    def load_model(self):
        pass
    
    def build_optimizer_scheduler(self):
        pass
    
    def build_noise_scheduler(self):
        pass
    
    def setup_pipeline(self):
        """Set up device assignment, environment, domain randomization, model, optimizer, learning rate scheduler. 
        Subclasses can override this if relevant setup depends on subclass-specific attributes that will be initialized later. """
        # Set up model, sim, and buffer devices. 
        self.set_devices()
        logger.info(colored("✓ Successfully set up model, simulation, and buffer devices", "green", "on_green"))
        
        # Create environment
        self.create_env()
        logger.info(colored("✓ Successfully created simulation environment", "green", "on_green"))
        if self.apply_domain_randomization:
            # more things to configure. define an object named self.domain_randomization, which has an apply() function. 
            self.build_domain_randomization()
            logger.info(colored("✓ Successfully built domain randomization", "green", "on_green"))
        
        # Load model (and optionally overload its config file)
        self.load_model()
        logger.info(colored("✓ Successfully loaded model", "green", "on_green"))
        
        ## Build learning rate scheduler, warmup scheduler, and optimizer. 
        ## Learning rate scheduler configuration
        self.lr_schedule = self.cfg.train.lr_schedule
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
                raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        ## Critic warmup configuration
        self.n_critic_warmup_itr = self.cfg.train.get("n_critic_warmup_itr", 0)  # Number of iterations to warmup critic
        self.critic_warmup_active = self.n_critic_warmup_itr > 0
        ## Actor and critic optimizer configuration (default: AdamW)
        self.actor_lr = self.cfg.train.actor_optimizer.lr 
        self.critic_lr = self.cfg.train.critic_optimizer.lr
        self.actor_betas = self.cfg.train.actor_optimizer.betas
        self.actor_eps   = self.cfg.train.actor_optimizer.eps
        self.actor_weight_decay = self.cfg.train.actor_optimizer.weight_decay
        self.critic_betas = self.cfg.train.critic_optimizer.betas
        self.critic_eps = self.cfg.train.critic_optimizer.eps
        self.critic_weight_decay = self.cfg.train.critic_optimizer.weight_decay
        self.build_optimizer_scheduler()
        logger.info(colored("✓ Successfully built optimizer and scheduler", "green", "on_green"))
        
        # Backup final configuration file (in case there are overloads before)
        save_config(self.cfg, Path(self.log_dir), 'cfg_rl')
        logger.info(colored("✓ Successfully saved configuration file", "green", "on_green"))