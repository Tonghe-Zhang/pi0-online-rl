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




import numpy as np
import torch
from typing import Dict, Any, Optional, List
import logging
logger=logging.getLogger(__name__)
import gc

class PiZeroSFTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for pi-zero SFT training, with single input and single output frames (no chunking). 
    
    This is a FLAT dataset where each item is a single timestep sample:
    {
        "observation.state": torch.Tensor,  # [state_dim]
        "action": torch.Tensor,             # [action_dim]
        "observation.images.top": torch.Tensor,  # [3, H, W]
        "task": str,                        # Task instruction
        "episode_index": int,               # Episode ID (for debugging)
        "frame_index": int,                 # Frame ID within episode
        "timestamp": float,                 # Timestamp
    }
    
    This structure is compatible with PyTorch DataLoader and pi-zero training.
    """
    
    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Args:
            samples: List of flat sample dictionaries
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class PiZeroSFTDatasetMultiStep(torch.utils.data.Dataset):
    """
    PyTorch Dataset for pi-zero SFT training with multi-step support.
    
    This dataset is inspired by the StitchedSequenceDataset from ReinFlow, adapted for PI0 data format.
    
    Key differences from StitchedSequenceDataset:
    - Uses episode_index instead of traj_lengths to determine episode boundaries
    - Works with PI0 sample format (observation.state, action, observation.images.top, task, etc.)
    - Maintains compatibility with existing PI0Policy training pipeline
    
    This dataset supports:
    - Multiple historical observations (cond_steps) for better conditioning
    - Multiple action outputs (horizon_steps / action chunking) for better temporal modeling
    - Proper episode boundary handling - repeats observations at episode start when needed
    - Skips episodes that are too short (< horizon_steps)
    
    Input format (flat samples):
    Each sample in the input list should be:
    {
        "observation.state": torch.Tensor,     # [state_dim]
        "action": torch.Tensor,                # [action_dim]
        "observation.images.top": torch.Tensor, # [3, H, W]
        "task": str,                           # Task instruction
        "episode_index": int,                  # Episode ID (0, 1, 2, ...)
        "frame_index": int,                    # Frame within episode
        "timestamp": float,                    # Timestamp
    }
    
    Output format (after __getitem__):
    {
        "observation.state": torch.Tensor,     # [cond_steps, state_dim] 
        "action": torch.Tensor,                # [horizon_steps, action_dim]
        "observation.images.top": torch.Tensor, # [img_cond_steps, 3, H, W]
        "task": str,                           # Task instruction
    }
    
    Episode boundary handling:
    - Like StitchedSequenceDataset, observations are repeated at episode boundaries
    - For example, if cond_steps=3 and we're at the first timestep of an episode,
      the observation history will be [obs[0], obs[0], obs[0]] where obs[0] is the first observation
    """
    
    def __init__(
        self, 
        samples: List[Dict[str, Any]], 
        horizon_steps: int = 4,
        cond_steps: int = 1,
        img_cond_steps: Optional[int] = None,
    ):
        """
        Args:
            samples: List of flat sample dictionaries ordered by episode_index
            horizon_steps: Number of future action steps to predict
            cond_steps: Number of historical observation steps to condition on
            img_cond_steps: Number of historical image steps (defaults to cond_steps)
        """
        self.samples = samples
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps if img_cond_steps is not None else cond_steps
        
        assert self.img_cond_steps <= self.cond_steps, "img_cond_steps should be <= cond_steps"
        
        # Build episode boundaries and indices
        self.episode_boundaries = self._build_episode_boundaries()
        self.indices = self._make_indices()
        
        logger.info(f"Created PiZeroSFTDatasetMultiStep with {len(self.indices)} valid sequences")
        logger.info(f"horizon_steps={horizon_steps}, cond_steps={cond_steps}, img_cond_steps={self.img_cond_steps}")
    
    def _build_episode_boundaries(self):
        """Build episode boundaries from episode_index"""
        if not self.samples:
            return []
        
        boundaries = []
        current_episode = self.samples[0]['episode_index']
        start_idx = 0
        
        for i, sample in enumerate(self.samples):
            episode_idx = sample['episode_index']
            if episode_idx != current_episode:
                # End of current episode
                boundaries.append((current_episode, start_idx, i))
                current_episode = episode_idx
                start_idx = i
        
        # Add the last episode
        boundaries.append((current_episode, start_idx, len(self.samples)))
        
        logger.info(f"Found {len(boundaries)} episodes")
        return boundaries
    
    def _make_indices(self):
        """
        Create indices for sampling, ensuring we don't cross episode boundaries.
        Each index is a tuple (start_idx, num_before_start) where:
        - start_idx: absolute index in self.samples where the sequence starts
        - num_before_start: number of steps before start_idx within same episode
        """
        indices = []
        
        for episode_idx, start_idx, end_idx in self.episode_boundaries:
            episode_length = end_idx - start_idx
            
            # Need at least horizon_steps for a valid sequence
            if episode_length < self.horizon_steps:
                continue
                
            # Create valid starting positions within this episode
            max_start = start_idx + episode_length - self.horizon_steps
            for i in range(start_idx, max_start + 1):
                num_before_start = i - start_idx
                indices.append((i, num_before_start))
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Return a multi-step sequence with proper boundary handling.
        """
        start_idx, num_before_start = self.indices[idx]
        end_idx = start_idx + self.horizon_steps
        
        # Get action sequence
        actions = []
        for i in range(start_idx, end_idx):
            actions.append(self.samples[i]['action'])
        actions = torch.stack(actions)  # [horizon_steps, action_dim]
        
        # Get state history with repetition at episode boundaries
        states = []
        for t in reversed(range(self.cond_steps)):
            state_idx = max(start_idx - t, start_idx - num_before_start)
            states.append(self.samples[state_idx]['observation.state'])
        states = torch.stack(states)  # [cond_steps, state_dim]
        
        # Get image history with repetition at episode boundaries
        images = []
        for t in reversed(range(self.img_cond_steps)):
            img_idx = max(start_idx - t, start_idx - num_before_start)
            images.append(self.samples[img_idx]['observation.images.top'])
        images = torch.stack(images)  # [img_cond_steps, 3, H, W]
        
        # Get task instruction (should be the same for the whole episode)
        task = self.samples[start_idx]['task']
        
        return {
            'observation.state': states,
            'action': actions,
            'observation.images.top': images,
            'task': task,
            'episode_index': self.samples[start_idx]['episode_index'],
            'frame_index': self.samples[start_idx]['frame_index'],
            'timestamp': self.samples[start_idx]['timestamp'],
        }


import threading
import json

class PiZeroSFTDatasetMultiStepIterable(torch.utils.data.Dataset):
    """
    Iterable multi-step dataset for sharded pi-zero SFT data.
    Loads only one shard at a time (or prefetches a few), never all at once.
    Supports large datasets split into many .pt files.

    Args:
        shards_metadata_path: Path to shards_metadata.json
        horizon_steps: Number of future action steps to predict
        cond_steps: Number of historical observation steps to condition on
        img_cond_steps: Number of historical image steps (defaults to cond_steps)
        num_workers: Number of background threads for prefetching (optional)
    """
    def __init__(self, shards_metadata_path: str, horizon_steps: int = 4, cond_steps: int = 1, img_cond_steps: Optional[int] = None, num_workers: int = 0):
        super().__init__()
        self.shards_metadata_path = shards_metadata_path
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps if img_cond_steps is not None else cond_steps
        self.num_workers = num_workers

        # Load metadata
        with open(shards_metadata_path, 'r') as f:
            meta = json.load(f)
        self.shards = meta['shards']
        self.num_shards = meta['num_shards']
        self.shard_size = meta['shard_size']
        self.total_episodes = meta['total_episodes']
        self.length = None # valid chunked samples
        self.image_key = meta.get('image_key', 'observation.images.top')
        if 'episode_lengths' not in meta:
            raise ValueError("Episode lengths not found in metadata. This will cause the length method to be very inefficient. Run the preprocessing script first!")
        self.episode_lengths = meta['episode_lengths']
        
        # Compute global sample offsets for each shard
        self.shard_offsets = []  # starting sample index for each shard
        offset = 0
        for shard in self.shards:
            self.shard_offsets.append(offset)
            offset += shard['num_samples']
        self.total_samples = offset

        # For each shard, build episode boundaries and indices (lazily)
        self._shard_indices = [None] * self.num_shards  # Will hold list of (start_idx, num_before_start) for each shard
        self._shard_lengths = [None] * self.num_shards  # Number of valid sequences in each shard
        self._shard_samples = [None] * self.num_shards  # Loaded samples (only one or a few in memory)
        self._shard_lock = threading.Lock()
        self._last_loaded = -1
        
        

    def _load_shard(self, shard_idx):
        shard_path = self.shards[shard_idx]['file']
        torch.serialization.add_safe_globals([np.core.multiarray.scalar]) # allow for numpy arrays to exist in the checkpoint
        torch.serialization.add_safe_globals([np.dtype]) # allow for numpy dtypes to exist in the checkpoint
        import os
        max_retries = 8
        for attempt in range(max_retries):
            try:
                # Verify file exists and has reasonable size
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Shard file not found: {shard_path}")
                file_size = os.path.getsize(shard_path)
                if file_size == 0:
                    raise ValueError(f"Shard file is empty: {shard_path}")
                # Try loading without mmap first
                samples = torch.load(shard_path, map_location='cpu', mmap=False, weights_only=False)
                # samples = torch.load(shard_path, map_location='cpu', mmap=True, weights_only=False)
                return samples
            except OSError as e:
                if "Input/output error" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"I/O error loading shard {shard_idx}, attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to load shard {shard_idx} after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                print(f"Unexpected error loading shard {shard_idx}: {e}")
                raise
            
    def _load_shard_unsafe(self, shard_idx):
        shard_path = self.shards[shard_idx]['file']
        torch.serialization.add_safe_globals([np.core.multiarray.scalar]) # allow for numpy arrays to exist in the checkpoint
        torch.serialization.add_safe_globals([np.dtype]) # allow for numpy dtypes to exist in the checkpoint
        samples = torch.load(shard_path, map_location='cpu', mmap=True, weights_only=False)
        return samples

    def _build_shard_indices(self, samples):
        # Copied from PiZeroSFTDatasetMultiStep logic
        if not samples:
            return [], 0
        boundaries = []
        current_episode = samples[0]['episode_index']
        start_idx = 0
        for i, sample in enumerate(samples):
            episode_idx = sample['episode_index']
            if episode_idx != current_episode:
                boundaries.append((current_episode, start_idx, i))
                current_episode = episode_idx
                start_idx = i
        boundaries.append((current_episode, start_idx, len(samples)))
        indices = []
        for episode_idx, start, end in boundaries:
            episode_length = end - start
            if episode_length < self.horizon_steps:
                continue
            max_start = start + episode_length - self.horizon_steps
            for i in range(start, max_start + 1):
                num_before_start = i - start
                indices.append((i, num_before_start))
        return indices, len(indices)

    def _ensure_shard_loaded(self, shard_idx):
        with self._shard_lock:
            if self._shard_samples[shard_idx] is None:
                samples = self._load_shard(shard_idx)
                indices, length = self._build_shard_indices(samples)
                self._shard_samples[shard_idx] = samples
                self._shard_indices[shard_idx] = indices
                self._shard_lengths[shard_idx] = length
                # Optionally unload previous shard to save memory
                if self._last_loaded != -1 and self._last_loaded != shard_idx:
                    self._shard_samples[self._last_loaded] = None
                    gc.collect()
                self._last_loaded = shard_idx
    
    # # Exactly compute the number of chunks in run-time. This is super inefficient and we abandon it. 
    # def __len__(self):
    #     # Total number of valid multi-step sequences across all shards
    #     if self.length is None:            
    #         total = 0
    #         for shard_idx in range(self.num_shards):
    #             if self._shard_lengths[shard_idx] is None:
    #                 # Load just the indices, not the samples
    #                 samples = self._load_shard(shard_idx)
    #                 indices, length = self._build_shard_indices(samples)
    #                 self._shard_indices[shard_idx] = indices
    #                 self._shard_lengths[shard_idx] = length
    #                 self._shard_samples[shard_idx] = None  # Don't keep samples in memory
    #                 del samples # free memory
    #             total += self._shard_lengths[shard_idx]
    #         self.length=total
    #     return self.length
    
    # Exactly compute the number of chunks from pre-processed files. 
    def __len__(self):
        if self.length is None:
            # Compute everything from episode_lengths without loading data
            episode_idx = 0
            total_valid_chunks = 0
            
            for shard_idx, shard_info in enumerate(self.shards):
                if self._shard_lengths[shard_idx] is None:
                    # Get this shard's episode lengths
                    shard_episode_count = shard_info['num_episodes']
                    shard_episodes = self.episode_lengths[episode_idx:episode_idx + shard_episode_count]
                    
                    # Build indices for this shard directly from episode lengths
                    indices = []
                    sample_idx = 0  # Position within this shard
                    
                    for ep_len in shard_episodes:
                        if ep_len >= self.horizon_steps:
                            # Add valid starting positions for this episode
                            for pos_in_episode in range(ep_len - self.horizon_steps + 1):
                                start_idx = sample_idx + pos_in_episode
                                num_before_start = pos_in_episode  # How many samples before episode start
                                indices.append((start_idx, num_before_start))
                        
                        sample_idx += ep_len  # Move to next episode's start
                    
                    # Cache the computed values
                    self._shard_indices[shard_idx] = indices
                    self._shard_lengths[shard_idx] = len(indices)
                    total_valid_chunks += len(indices)
                    
                    episode_idx += shard_episode_count
            
            self.length = total_valid_chunks
            print(f"All shard indices computed from episode_lengths. Total: {self.length}")
        
        return self.length
        
    
    def _find_shard_and_local_idx(self, global_idx):
        # Find which shard and which index within that shard
        count = 0
        for shard_idx in range(self.num_shards):
            length = self._shard_lengths[shard_idx]
            if length is None:
                # # Load just the indices
                # samples = self._load_shard(shard_idx)
                # indices, length = self._build_shard_indices(samples)
                # self._shard_indices[shard_idx] = indices
                # self._shard_lengths[shard_idx] = length
                # self._shard_samples[shard_idx] = None
                raise ValueError(f"shard_lengths is not initialized in __len__. ")
            if global_idx < count + length:
                local_idx = global_idx - count
                return shard_idx, local_idx
            count += length
        raise IndexError(f"Index {global_idx} out of range (total {count})")

    def __getitem__(self, idx):
        # logger.info(f"_find_shard_and_local_idx:")
        shard_idx, local_idx = self._find_shard_and_local_idx(idx)
        # logger.info(f"_find_shard_and_local_idx finished.")

        # logger.info(f"_ensure_shard_loaded")
        self._ensure_shard_loaded(shard_idx)
        # logger.info(f"_ensure_shard_loaded finished.")
        
        samples = self._shard_samples[shard_idx]
        indices = self._shard_indices[shard_idx]
        # logger.info(f"samples={samples}, indices={indices}")
        start_idx, num_before_start = indices[local_idx]
        end_idx = start_idx + self.horizon_steps
        # Get action sequence
        actions = [samples[i]['action'] for i in range(start_idx, end_idx)]
        actions = torch.stack(actions)
        # Get state history
        states = [samples[max(start_idx - t, start_idx - num_before_start)]['observation.state'] for t in reversed(range(self.cond_steps))]
        states = torch.stack(states)
        # Get image history
        images = [samples[max(start_idx - t, start_idx - num_before_start)][self.image_key] for t in reversed(range(self.img_cond_steps))]
        images = torch.stack(images)
        task = samples[start_idx]['task']
        return {
            'observation.state': states,
            'action': actions,
            'observation.images.top': images,
            'task': task,
            'episode_index': samples[start_idx]['episode_index'],
            'frame_index': samples[start_idx]['frame_index'],
            'timestamp': samples[start_idx]['timestamp'],
        }

    # Optionally: implement background prefetching with threads/queues for num_workers > 0 (not shown here for brevity). This will require us to modify the threading lock logic. 

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='/nvme_data/tonghe/RL4VLA/datasets/mp_collect/PutOnPlateInScene25Main-v3/12800/shards-split/train_dataset_sharded.json')
    parser.add_argument("--val_path", type=str, default='/nvme_data/tonghe/RL4VLA/datasets/mp_collect/PutOnPlateInScene25Main-v3/12800/shards-split/val_dataset_sharded.json')
    parser.add_argument("--horizon_steps", type=int, default=4)
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--img_cond_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_sample_number", type=int, default=1798)
    parser.add_argument("--val_sample_number", type=int, default=254)
    args = parser.parse_args()
    logger.info(f"Build PiZeroSFTDatasetMultiStepIterable from training shards metadata {args.train_path}")
    train_dataset = PiZeroSFTDatasetMultiStepIterable(shards_metadata_path=args.train_path, horizon_steps=args.horizon_steps, cond_steps=args.cond_steps, img_cond_steps=args.img_cond_steps, num_workers=args.num_workers)
    logger.info(f"Build PiZeroSFTDatasetMultiStepIterable from validation shards metadata {args.val_path}")
    val_dataset = PiZeroSFTDatasetMultiStepIterable(shards_metadata_path=args.val_path, horizon_steps=args.horizon_steps, cond_steps=args.cond_steps, img_cond_steps=args.img_cond_steps, num_workers=args.num_workers)
    logger.info(f"train_dataset length = {len(train_dataset)}")  # 344473
    logger.info(f"train_dataset[{args.train_sample_number}] = {train_dataset[args.train_sample_number]}")
    for i in range(10):
        logger.info(f"train_dataset[{i}] = {train_dataset[args.train_sample_number+i]}")
    
    logger.info(f"val_dataset length = {len(val_dataset)}")      #  18082
    logger.info(f"val_dataset[{args.val_sample_number}] = {val_dataset[args.val_sample_number]}")
    for i in range(10):
        logger.info(f"val_dataset[{i}] = {val_dataset[args.val_sample_number+i]}")