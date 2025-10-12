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


# MIT License

# Copyright (c) 2024 simpler-env

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

# Revised by Tonghe on 2025-07-14. 

import gymnasium
from mani_skill.envs.sapien_env import BaseEnv
class PerStepRewardWrapper(gymnasium.Wrapper):
    """
    A wrapper that makes GPU parallelized environments that only provides episodic success flags (e.g. SimplerEnv) 
    able to return per-step sparse reward according to the success flags. 
    Currently we do not support dense reward, or fail flags. 
    """
    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.env: BaseEnv = env
        self.num_envs = env.unwrapped.num_envs # type: ignore
        self.debug_mode=False #True  # for development only.  
     
    def reset(self, seed=None, options: dict = {})->tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info # type: ignore
    
    def step(self, action_venv):
        """
        Wraps the ManiSkill3 step function to provide per-step reward from episode success flags. 
        Use case: 
            nextobs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.env.step(action_venv)
        """
        # import logging 
        # logger=logging.getLogger(__name__)
        # logger.info(f"*{self.__class__.__name__}: step()")
        # the _reward_venv was always 0, since reward_mode=None. But we need to overwrite it with the per-step reward. 
        nextobs_venv, _reward_venv, _terminated_venv, truncated_venv, info_venv = self.env.step(action_venv)  # original per step reward is always 0, since reward_mode=None. 

        # Derive per-step reward from episode success flags. 
        # info_venv: dict. Additional information from the environment, this one could be very large and contains many different types of information. 
        # For PutOnPlateInScene tasks, it contains, keys=['elapsed_steps', 'is_src_obj_grasped', 'consecutive_grasp', 'src_on_target', 'gripper_carrot_dist', 'gripper_plate_dist', 'carrot_plate_dist', 'success', 'reconfigure']
        # and info['success'] is a tensor of shape [num_envs,], dtype=bool. 
        # We only need to use info['success'] to derive per-step sparse reward, while other information are also needed to derive a dense reward (though deprecated). 
        
        
        # About the relation between success and reward. 
        # In ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene_multi.py, 
        # the success is defined as the object is on the target plate with a minimal contact force. 
        # however, this success flag does not emphasize the success when the source object is grasped. 
        # so the learner may hack the reward by throwing the objects to the plate and get a high reward, but this is not a desired behavior. 
        # There are two ways to correct this. 
        # First, use reward_venv = info_venv["success"] & info_venv["is_src_obj_grasped"] # [num_envs]
        # Second, use reward_venv = info_venv["success"] and revise the evaluate() function in the environment file to define success = previous_success_definition & is_src_obj_grasped. 
        # The two methods are not completely equivalent -- although for each episode they are the same, the second approach 
        # completely changes the success definition, which may alter the way `terminated` is defined and affect partial reset and episode numbers. 
        # The first approach will terminate an episode whent he robot throws the object and receice 0 reward, 
        # the second approach will not terminate the episode, and encourage the robot to pick it up and place it. 
        # The second approach will make we use spare-reward, but may enlongate episode length or cause more truncations, and reduce the success rate  (e.g. 15% --> 13.4%) than the first approach. 
        # The first approach is actually using reward-shaping, which is not strictly a sparse reward. 
        # To use sparse reward, we use the second approach. 
        
        reward_venv = info_venv["success"] # [num_envs]
        
        #################################################
        if self.debug_mode: 
            import logging 
            logger=logging.getLogger(__name__)
            logger.info(f"DEBUG::PerStepRewardWrapper: terminated={_terminated_venv}, info['success']={info_venv['success']}")
            logger.info(f"DEBUG::PerStepRewardWrapper: info_venv['is_src_obj_grasped']={info_venv['is_src_obj_grasped']}, reward_venv={reward_venv}")
            if reward_venv.sum() > 0:
                import torch
                from termcolor import colored
                reward_venv: torch.Tensor
                logger.info(colored(f"DEBUG::PerStepRewardWrapper: positive reward detected at {torch.nonzero(reward_venv > 0, as_tuple=False)[:, 0].tolist()}", "blue", "on_blue"))
        return nextobs_venv, reward_venv, _terminated_venv, truncated_venv, info_venv

    def close(self):
        self.env.close()