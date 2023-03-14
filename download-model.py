'''
Downloads models from Hugging Face to models/model-name.

Example:
python download-model.py facebook/opt-1.3b

Taken from  https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py
'''

import argparse
import base64
import json
import multiprocessing
import re
import sys
from pathlib import Path

import requests
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str, default=None, nargs='?')
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('--threads', type=int, default=1, help='Number of files to download simultaneously.')
parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
args = parser.parse_args()

def get_file(args):
    url = args[0]
    output_folder = args[1]
    idx = args[2]
    tot = args[3]

    print(f"Downloading file {idx} of {tot}...")
    r = requests.get(url, stream=True)
    with open(output_folder / Path(url.split('/')[-1]), 'wb') as f:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
        t.close()

def sanitize_branch_name(branch_name):
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if pattern.match(branch_name):
        return branch_name
    else:
        raise ValueError("Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

def get_download_links_from_huggingface(model, branch):
    base = "https://huggingface.co"
    page = f"/api/models/{model}/tree/{branch}?cursor="
    cursor = b""

    links = []
    classifications = []
    has_pytorch = False
    has_safetensors = False
    while True:
        content = requests.get(f"{base}{page}{cursor.decode()}").content

        dict = json.loads(content)
        if len(dict) == 0:
            break

        for i in range(len(dict)):
            fname = dict[i]['path']

            is_pytorch = re.match("pytorch_model.*\.bin", fname)
            is_safetensors = re.match("model.*\.safetensors", fname)
            is_tokenizer = re.match("tokenizer.*\.model", fname)
            is_text = re.match(".*\.(txt|json)", fname) or is_tokenizer

            if any((is_pytorch, is_safetensors, is_text, is_tokenizer)):
                if is_text:
                    links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                    classifications.append('text')
                    continue
                if not args.text_only:
                    links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                    if is_safetensors:
                        has_safetensors = True
                        classifications.append('safetensors')
                    elif is_pytorch:
                        has_pytorch = True
                        classifications.append('pytorch')

        cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
        cursor = base64.b64encode(cursor)
        cursor = cursor.replace(b'=', b'%3D')

    # If both pytorch and safetensors are available, download safetensors only
    if has_pytorch and has_safetensors:
        for i in range(len(classifications)-1, -1, -1):
            if classifications[i] == 'pytorch':
                links.pop(i)

    return links

if __name__ == '__main__':
    model = args.MODEL
    branch = args.branch
    if model is None:
        model, branch = select_model_from_default_options()
    else:
        if model[-1] == '/':
            model = model[:-1]
            branch = args.branch
        if branch is None:
            branch = "main"
        else:
            try:
                branch = sanitize_branch_name(branch)
            except ValueError as err_branch:
                print(f"Error: {err_branch}")
                sys.exit()
    if branch != 'main':
        output_folder = Path("models") / (model.split('/')[-1] + f'_{branch}')
    else:
        output_folder = Path("models") / model.split('/')[-1]
    if not output_folder.exists():
        output_folder.mkdir()

    links = get_download_links_from_huggingface(model, branch)

    # Downloading the files
    print(f"Downloading the model to {output_folder}")
    pool = multiprocessing.Pool(processes=args.threads)
    results = pool.map(get_file, [[links[i], output_folder, i+1, len(links)] for i in range(len(links))])
    pool.close()
    pool.join()
