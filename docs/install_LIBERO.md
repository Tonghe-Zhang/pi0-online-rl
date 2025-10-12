



# About this file
This file walks you through the steps to install mujoco_py, robosuite, robomimic, and eventually the LIBERO simulation framework in the pi-zero environment.



## Install mujoco_py, robosuite, robomimic
Here we list one possible approach to install mujoco_py. 

```bash
# make mujoco directory
mkdir $HOME/.mujoco
cd ~/.mujoco 
# download mujoco210 and mujoco-py from this source, or other places you like
https://drive.google.com/drive/folders/15fcrWlTCwFxZkxpMPSrcNzCfTxjj8WE4?usp=sharing
# unzip them to your root directory
unzip ./mujoco210.zip
unzip ./mujoco-py.zip
# add link to mujoco
echo -e '# link to mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" ' >> ~/.bashrc
source ~/.bashrc
conda activate reinflow
# install cython
pip install 'cython<3.0.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

# if you don't have root privilege or cannot update the driver (e.g. in a container)
pip install patchelf
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
# else, if you have root privilege: 
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# install mujoco-py
cd ~/.mujoco/mujoco-py
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e . --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple
# test mujoco-py installation 
python3
import mujoco_py # there should be no import errors. 
dir(mujoco_py)   # you should see a lot of methods if you successfully installed mujoco_py. 
```

* **[Debug Help]**
If you don't have root privilege, or meet the error '#include <GL/glew.h>'...
```bash
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
You can still install mujoco without using sudo commands by the following bash commands, according to https://github.com/openai/mujoco-py/issues/627
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
```

* **[Debug Help]**
If you meet this error: version `GLIBCXX_3.4.30' not found...
```bash
# link GLIBCXX_3.4.30. reference: https://blog.csdn.net/L0_L0/article/details/129469593
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX # 1. first check if the thing exists: 
# you should be ablt to see GLIBCXX_3.4 ~ GLIBCXX_3.4.30
cd <PATH_TO_YOUR_ANACONDA3>/envs/reinflow/bin/../lib/ # 2. then create soft links
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

* **[Debug Help]**
If you meet this error: FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'
```python
pip install patchelf
```



## Install LIBERO
For those operating on a linux machine, first do:
```bash 
# install cmake<4.0.0
pip install cmake==3.31.6
# download egl_prob from another branch: (ensure internet connection)
wget https://github.com/mhandb/egl_probe/archive/fix_windows_build.zip
# install egl_prob from another branch:
pip install fix_windows_build.zip
```
> This will help you prevent the error 
> ```bash
> raise RuntimeError("CMake must be installed.")
> ```

Then, follow the [instructions](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Installation) in LIBERO's official repository to install LIBERAO:
```bash 
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt  # install LIBERO dependencies.
pip install -e .  # install the libero package.
```

Next, download the all the four suites of LIBERO human teleoperation datasets to /LIBERO/libero/datasets, 
```bash
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```
Beware of the space consumption of these data. The four suites take up around of 34.9 GB disk space. 
You may specify other data directories or a subset of LIBERO suites following the official repository's instructions. 

If downloading `libero_100` from huggingface fails, try switching to the original links:
```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```