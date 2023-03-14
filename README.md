# llama-7b-hf Tuning with [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) Dataset using Deepspeed and Transformers

This is my first go at ML tuning, so this is probably very wrong. This should work on a single ~~3090 GPU~~ A100 and takes 3 hours to train 250 setps on a subset of 1000 samples. Full 50k~ dataset should take ~19 hours. There's a lot of knobs to turn that I don't understand yet.

**I'm currently running this, so I don't know if it even works**

References:
 - https://github.com/tatsu-lab/stanford_alpaca#data-release
 - https://github.com/facebookresearch/llama/issues/169
 - https://github.com/huggingface/transformers/pull/21955 

Prereqs
```
conda install -y cuda pytorch-cuda=11.7 -c pytorch -c nvidia
```

Run tuning
```
git clone https://github.com/tatsu-lab/stanford_alpaca repositories/stanford_alpaca
pip install -r requirements.txt
python download-model.py decapoda-research/llama-7b-hf
deepspeed tune.py
```
