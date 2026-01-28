"""
compute thresholds for medical SAE features from gemma 9b
"""

import os
import sys
import argparse
import pickle
import requests

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
parser.add_argument('--explanation_model', type=str, default='claude-3-5-haiku-20241022')
args = parser.parse_args()

neuronpedia_api_key = os.environ.get('NEURONPEDIA_API_KEY')

thresholds = {}
def get_threshold(feature_idx, sae_layer, api_key=None):
    """
    get threshold for a given feature from neuronpedia
    """

    # first check if already computed, just return
    if feature_idx in thresholds:
        return thresholds[feature_idx]
    
    model_name = "gemma-2-9b"
    sae_name = f"{sae_layer}-gemmascope-res-16k"  # canonical width

    # second, try to get from neuronpedia
    url = f"https://www.neuronpedia.org/api/feature/{model_name}/{sae_name}/{feature_idx}"
    response = requests.get(url, timeout=30)
    acts = response.json()['activations']

    all_bins = set()
    for i in range(len(acts)):
        all_bins.add(acts[i]['binMax'])
    
    all_bins = list(all_bins)
    all_bins.sort(reverse=True)    
    thresholds[feature_idx] = all_bins
    return all_bins

scores = {}
def get_score(feature_idx, sae_layer, api_key=None):
    """
    get score for a given feature from neuronpedia
    """
    # first check if already computed, just return
    if feature_idx in scores:
        return scores[feature_idx]
    
    model_name = "gemma-2-9b"
    sae_name = f"{sae_layer}-gemmascope-res-16k"  # canonical width

    # second, try to get from neuronpedia

    # first we need to get the explanation id
    url = f"https://www.neuronpedia.org/api/feature/{model_name}/{sae_name}/{feature_idx}"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        explanations = response.json()['explanations']
        for explanation in explanations:
            if explanation.get('typeName') == 'oai_token-act-pair' and explanation.get('explanationModelName') == args.explanation_model:
                for score in explanation.get('scores'):
                    if score.get('explanationScoreTypeName') == 'eleuther_recall':
                        scores[feature_idx] = score.get('value')
                        return score.get('value')
                explanationId = explanation.get('id')
                break   
    
    # then we can use this to get a score    
    url = f"https://www.neuronpedia.org/api/explanation/score"
    response = requests.post(url, timeout=30,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key
        },
        json={
            "explanationId": explanationId,
            "scorerModel": "claude-3-5-haiku-20241022",
            "scorerType": "eleuther_recall"
        })
    
    if response.status_code == 200:
        scores[feature_idx] = response.json()['score']['value']
        return response.json()['score']['value']
    else:
        print(f"Error getting score for feature {feature_idx}: {response.status_code} | {response.text}")
        return None
    

df = pd.read_csv('gemma-9b-l31-16k-descriptions-haiku-medical.csv')
results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc='computing thresholds'):
    results.append({
        'idx': row['feature_idx'],
        'explanation': row['feature_description'],
        'thresholds': get_threshold(row['feature_idx'], 31, neuronpedia_api_key),
        'score': get_score(row['feature_idx'], 31, neuronpedia_api_key)
    })

df = pd.DataFrame(results)
df.to_json('features.jsonl', orient='records', lines=True)
print('wrote to features.jsonl')