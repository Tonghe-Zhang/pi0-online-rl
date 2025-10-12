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
import os 

def make_dataset_stats(dataset_path:str):
    for episode_path in os.listdir(dataset_path):
        if episode_path.endswith('.npz'):
            with np.load(os.path.join(dataset_path,episode_path), allow_pickle=True) as data:
                arr_0 = data['arr_0'].item()
                print(f"arr_0={arr_0.keys()}")
                print(f"is_image_encode: {arr_0['is_image_encode']}")
                print(f"instruction: {arr_0['instruction']}")
                print(f"action: {arr_0['action'].shape}")
                print(f"image: {len(arr_0['image'])}")
                print(f"info: len={len(arr_0['info'])}, in each frame, {(arr_0['info'][0].keys())}")
                exit()
                # print(f"image: {arr_0['image']}")





if __name__=="__main__":
    dataset_path='data/ManiSkill3/PutOnPlateInScene25Single-v1/data/'
    make_dataset_stats(dataset_path)
    