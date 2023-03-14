# llama-7b-hf Tuning with [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) Dataset using Deepspeed and Transformers

This is my first go at ML tuning, so this is probably very wrong. This should work on a single 3090 GPU and takes 3 hours. There's a lot of knobs tu turn that I don't understand yet.

References:
 - https://github.com/tatsu-lab/stanford_alpaca#data-release
 - https://github.com/facebookresearch/llama/issues/169
 - https://github.com/huggingface/transformers/pull/21955 

Prereqs
```
conda install python=3.10 pip --force-reinstall
conda -y install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

Run tuning
```
pip install -r requirements.txt
python download-model.py decapoda-research/llama-7b-hf
deepspeed tune.py
```
