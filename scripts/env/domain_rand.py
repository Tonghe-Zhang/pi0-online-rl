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



"""
Apply domain randomization to the input batch. 
Currently we only support visual augmentation. 
"""
import random
from typing import List
from termcolor import colored
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import logging
logger = logging.getLogger(__name__)

class DomainRandomization:
    def __init__(self, domain_rand_cfg):
        # Currently we only support visual augmentation. In the future we will support other modalities, including text and proprioception. 
        self.domain_rand_cfg = domain_rand_cfg if domain_rand_cfg is not None else {}
        
        self.visual_augment_list:dict=domain_rand_cfg.get('visual_augment_type', {})
        self.visual_augment_order:List[str]=domain_rand_cfg.get('visual_augment_order', [])
        self.debug_mode = domain_rand_cfg.get('debug_mode', False)
        if self.domain_rand_cfg=={} or (self.visual_augment_list=={}) or (self.visual_augment_order==[]):
            self.empty_mode=True
            logger.warning("Domain randomization configuration is empty. Please check your config file.")
        else:
            self.empty_mode=False
    
    def __repr__(self):
        return f"DomainRandomization(visual_augment_list={self.visual_augment_list}, visual_augment_order={self.visual_augment_order}, debug_mode={self.debug_mode}, empty_mode={self.empty_mode})"
    
    
    @torch.no_grad
    def apply(self, batch:dict):
        if self.empty_mode:
            return batch
        if self.debug_mode:
            print(colored(f"DEBUG:: domain_randomization applied", "red", "on_red"))
        # Visual augmentation
        # Find all image keys (4D tensors, e.g. [B, C, H, W])
        image_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor) and v.ndim == 4]
        for aug_name in self.visual_augment_order:
            if self.debug_mode: print(colored(f"DEBUG:: domain_randomization {aug_name} applied", "red", "on_red"))
            for k in image_keys:
                if self.debug_mode: print(colored(f"DEBUG:: domain_randomization {aug_name} applied to {k}", "red", "on_red"))
                img = batch[k]
                if aug_name == "random_resized_crop":
                    scale = self.visual_augment_list[aug_name].get("scale", [0.9, 0.9])
                    ratio = self.visual_augment_list[aug_name].get("ratio", [1.0, 1.0])
                    imgs = []
                    for i in range(img.shape[0]):
                        params = RandomResizedCrop.get_params(
                            img[i], scale=list(scale), ratio=list(ratio)
                        )
                        cropped = F.resized_crop(img[i], *params, img.shape[2:], interpolation=InterpolationMode.BILINEAR)
                        imgs.append(cropped)
                    batch[k] = torch.stack(imgs)
                elif aug_name == "random_brightness":
                    brightness = self.visual_augment_list[aug_name][0] if isinstance(self.visual_augment_list[aug_name], list) else self.visual_augment_list[aug_name]
                    imgs = [F.adjust_brightness(img[i], 1.0 + random.uniform(-brightness, brightness)) for i in range(img.shape[0])]
                    batch[k] = torch.stack(imgs)
                elif aug_name == "random_contrast":
                    contrast = self.visual_augment_list[aug_name]
                    if isinstance(contrast, dict) or (hasattr(contrast, 'min') and hasattr(contrast, 'max')):
                        min_c = float(contrast.get('min', 1.0))
                        max_c = float(contrast.get('max', 1.0))
                    elif isinstance(contrast, list) or isinstance(contrast, tuple):
                        min_c, max_c = float(contrast[0]), float(contrast[1])
                    else:
                        min_c = max_c = float(contrast)
                    imgs = [F.adjust_contrast(img[i], random.uniform(min_c, max_c)) for i in range(img.shape[0])]
                    batch[k] = torch.stack(imgs)
                elif aug_name == "random_saturation":
                    saturation = self.visual_augment_list[aug_name]
                    if isinstance(saturation, dict) or (hasattr(saturation, 'min') and hasattr(saturation, 'max')):
                        min_s = float(saturation.get('min', 1.0))
                        max_s = float(saturation.get('max', 1.0))
                    elif isinstance(saturation, list) or isinstance(saturation, tuple):
                        min_s, max_s = float(saturation[0]), float(saturation[1])
                    else:
                        min_s = max_s = float(saturation)
                    imgs = [F.adjust_saturation(img[i], random.uniform(min_s, max_s)) for i in range(img.shape[0])]
                    batch[k] = torch.stack(imgs)
                elif aug_name == "random_hue":
                    hue = self.visual_augment_list[aug_name][0] if isinstance(self.visual_augment_list[aug_name], list) else self.visual_augment_list[aug_name]
                    imgs = [F.adjust_hue(img[i], random.uniform(-hue, hue)) for i in range(img.shape[0])]
                    batch[k] = torch.stack(imgs)
                else:
                    raise NotImplementedError(f"Augmentation {aug_name} not implemented.")
        return batch
        
    
    
    