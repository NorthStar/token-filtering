"""
get explanations for Gemma 2 9b layer 20 SAE features
"""

import os
import sys
import argparse
import pickle
import requests
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sympy.printing import preview
from tqdm import tqdm


import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

import tiktoken
from sae_lens import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='google/gemma-2-9b')
parser.add_argument("--explanation_model", type=str, default='gpt-4o-mini')
parser.add_argument("--medical_features", action='store_true')
args = parser.parse_args()

neuronpedia_api_key = os.environ.get('NEURONPEDIA_API_KEY')

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model
)

sae_layer = 31
sae_id = "gemma-scope-9b-pt-res-canonical"
print(f"loading {sae_id} sae @ layer {sae_layer}")

sae = SAE.from_pretrained(
    release=sae_id,
    sae_id=f"layer_{sae_layer}/width_16k/canonical",
    device=model.device
)
print(f"sae has {sae.cfg.d_sae} features")

feature_descriptions = {}
def get_feature_description(feature_idx, sae_layer, api_key=None):
    """
    get description from neuronpedia
    """

    # first check if already computed, just return
    if feature_idx in feature_descriptions:
        return feature_descriptions[feature_idx]
            
    model_name = "gemma-2-9b"
    sae_name = f"{sae_layer}-gemmascope-res-16k"  # Assuming 16k width

    # second try to get from neuronpedia if it's already been computed there
    try:
        url = f"https://www.neuronpedia.org/api/feature/{model_name}/{sae_name}/{feature_idx}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            explanations = response.json()['explanations']
            for explanation in explanations:
                if explanation.get('typeName') == 'oai_token-act-pair' and explanation.get('explanationModelName') == args.explanation_model:
                    feature_descriptions[feature_idx] = explanation.get('description')
                    return feature_descriptions[feature_idx]
    except:
        pass
    
    # third, if not, we'll need to generate it w/ haiku
    url = f"https://www.neuronpedia.org/api/explanation/generate"
    payload = {
        'modelId' : model_name,
        'layer' : sae_name,
        'index' : feature_idx,
        'explanationType' : 'oai_token-act-pair',
        'explanationModelName' : args.explanation_model
    }
    
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
    
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 500:
        print(f"error getting feature description for feature {feature_idx}: {response.status_code} | {response.text}")
        time.sleep(10)
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"Error getting feature description for feature {feature_idx}: {response.status_code} | {response.text}")
            return ''
        else:
            feature_descriptions[feature_idx] = response.json()['explanation']['description']
            return feature_descriptions[feature_idx]
    if response.status_code != 200:
        print(f"Error getting feature description for feature {feature_idx}: {response.status_code} | {response.text}")
        return ''
    else:
        feature_descriptions[feature_idx] = response.json()['explanation']['description']
        return feature_descriptions[feature_idx]

if args.medical_features:
    df = pd.read_csv('gemma-9b-l31-16k-descriptions-4o-mini-medical.csv')
    feature_ids = df['feature_idx'].tolist()
else:
    feature_ids = list(range(sae.cfg.d_sae))

inputs = tokenizer(
    'The quick brown fox jumps over the lazy dog',
    return_tensors="pt",
    max_length=512,
    padding=True,
    truncation=True
).to(model.device)

with torch.no_grad():

    outputs = model(**inputs, output_hidden_states=True)
    attn_output = outputs.hidden_states[sae_layer] # (batch_size, seq_len, n_embd)
    first_token_attn = attn_output[:, 0, :] # (batch_size, n_embd)

    sae_output = sae.encode(first_token_attn) # [batch_size, n_sae_features]
    print(f"getting explanations for {len(feature_ids)} SAE features...")
    
    for feature_idx in tqdm(feature_ids):
        feature_description = get_feature_description(feature_idx, sae_layer, neuronpedia_api_key)

print(len(feature_descriptions))
df = pd.DataFrame(list(feature_descriptions.items()), columns=['feature_idx', 'feature_description'])
print(df.head())

if args.explanation_model == 'gpt-4o-mini':
    explanation_model = '4o-mini'
if args.explanation_model == 'gpt-4o':
    explanation_model = '4o'
elif args.explanation_model == 'claude-sonnet-3-7-20250219':
    explanation_model = 'sonnet'
else:
    explanation_model = args.explanation_model

if args.medical_features:
    df.to_csv(f'gemma-9b-l31-16k-descriptions-{explanation_model}-medical.csv', index=False)
else:
    df.to_csv(f'gemma-9b-l31-16k-descriptions-{explanation_model}.csv', index=False)