# llama-7b-hf Tuning with [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) Dataset using Deepspeed and Transformers

This is my first go at ML tuning, so this is probably very wrong. This should work on a single ~~3090 GPU~~ A100 and takes 3 hours to train 250 setps on a subset of 1000 samples. Full 50k~ dataset should take ~19 hours. There's a lot of knobs to turn that I don't understand yet.

Wishlist:

- [ ] Command Flags (Currently everything is in code)
- [ ] Auto Tuning 
- [X] Windows Support (Even with DeepSpeed!)

**I'm currently running this, so I don't know if it even works**

References:
 - https://github.com/tatsu-lab/stanford_alpaca#data-release
 - https://github.com/facebookresearch/llama/issues/169
 - https://github.com/huggingface/transformers/pull/21955 

### Prereqs

Installing pytorch and cuda is the hardest part of machine learning
I've come up with this install line from the following sources:

- https://pytorch.org/get-started/locally/#start-locally
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-previous-cuda-releases

```
conda install -y cuda pybind pytorch=1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia/label/cuda-11.7.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -c 'import torch; print(torch.cuda.is_available())'
```

For Windows, you'll need to compile DeepSpeed (which actually works!)

Install [Visual Studio 2019 Build Tools](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers). Click on the latest **BuildTools** link, Select **Desktop Environment with C++** when installing

In an **Administrator Powershell**, do:

```
git clone https://github.com/microsoft/DeepSpeed.git repositories/DeepSpeed
cd repositories/DeepSpeed
.\build_win.bat
pip install (Get-Item .\dist\*)
```

Get data, model, and requirements:

```
git clone https://github.com/tatsu-lab/stanford_alpaca repositories/stanford_alpaca
pip install -r requirements.txt
python download-model.py decapoda-research/llama-7b-hf
```

Run tuning:

```
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
deepspeed tune.py
```

Or, for Windows Powershell:
```
$env:CUDA_VISIBLE_DEVICES=0
python (Get-Command ds).Path tune.py
``` 

Comparing the model tuned on 250-step 1000 training samples VS vanilla llama-7b:

![image](https://user-images.githubusercontent.com/1486609/224945013-8c7a1942-660d-41f3-b659-4baa055a0d1e.png)

Vanilla llama-7b:

![image](https://user-images.githubusercontent.com/1486609/224945042-c82da755-d2a2-480e-bf60-367f159dbbbc.png)
