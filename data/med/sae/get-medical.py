import argparse
import os
import asyncio
import sys
import logging
import pandas as pd
from tqdm.asyncio import tqdm
from pathlib import Path
import requests
import random

logging.getLogger('safetytooling').setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
safety_tooling_path = os.path.join(parent_dir, "safety-tooling")
sys.path.append(safety_tooling_path)

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompt.txt")
parser.add_argument("--max_concurrents", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1600)
parser.add_argument("--input_file", type=str, default="gemma-9b-l31-16k-descriptions-4o-mini.csv")
parser.add_argument("--examples", action='store_true')
args = parser.parse_args()

neuronpedia_api_key = os.environ.get('NEURONPEDIA_API_KEY')

class Classifier:
    def __init__(self, max_concurrents = 5, prompt_file = args.prompt):

        """initialize the classifier with safety-tooling API."""
        self.max_concurrents = max_concurrents
        self.semaphore = asyncio.Semaphore(max_concurrents)
        
        # setup environment for API keys
        utils.setup_environment()
        
        # initialize the inference API
        self.api = InferenceAPI(
            cache_dir=Path(".cache"),
            anthropic_num_threads=max_concurrents
        )
        
        # load system prompt from file or use default
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
        print(f"loaded prompt from {prompt_file}")

    async def classify_sentence(self, feature_description, examples=None):

        # classify a single document
        async with self.semaphore:

            try:

                # create the prompt
                if examples:
                    prompt = Prompt(messages= [
                        ChatMessage(content=self.system_prompt, role=MessageRole.system),
                        ChatMessage(content=f'Description: {feature_description}\nExamples: {examples}', role=MessageRole.user)
                    ])
                else:
                    prompt = Prompt(messages= [
                        ChatMessage(content=self.system_prompt, role=MessageRole.system),
                        ChatMessage(content=f'Description: {feature_description}', role=MessageRole.user)
                    ])
                
                # make API call
                response = await self.api(
                    model_id="claude-sonnet-4-20250514",
                    prompt=prompt,
                    max_attempts_per_api_call=3,
                    n=1
                )
                
                # parse response
                if response and len(response) > 0:
                    answer = response[0].completion.strip().upper()
                    is_medical = answer.startswith("YES")
                    return is_medical
                else:
                    print(f"warning: no response for document")
                    return False
                    
            except Exception as e:

                print(f"error classifying document: {e}")
                return False
    
    async def process_sentences(self, sentences):

        batch_size = self.max_concurrents * 2 # process more than concurrent limit to keep queue full
        annotated = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            print(f"processing batch {i//batch_size + 1}, sentences {i}-{min(i+batch_size, len(sentences))}")
            
            # create tasks for this batch
            tasks = []
            for sentence in batch:
                if args.examples:
                    sent_examples = list(set(sentence['examples']))
                    examples_text = ', '.join(random.sample(sent_examples, min(10, len(sent_examples))))
                    tasks.append(self.classify_sentence(sentence['feature_description'], examples_text.replace('_', ' ')))
                else:
                    tasks.append(self.classify_sentence(sentence['feature_description']))
            
            # process batch concurrently
            print(examples_text)
            results = await tqdm.gather(*tasks, desc="classifying sentences")

            for sentence, is_medical in zip(batch, results):
                if is_medical:
                    annotated.append({
                        'feature_idx': sentence['feature_idx'],
                        'feature_description': sentence['feature_description']
                    })
            
        return annotated
    
# helper function to get contiguous tokens above threshold
def get_contiguous_tokens(tokens, values, threshold):
    result = []
    current = []
    current_over_threshold = False
    
    for token, value in zip(tokens, values):
        if value > threshold:
            current_over_threshold = True
        
        if value > 0.0:
            current.append(token.replace('_', ' '))        
        elif current and current_over_threshold:
            result.append(''.join(current).replace('_', ' '))
            current = []
            current_over_threshold = False
        elif current:
            current = []
            current_over_threshold = False
    
    if current:  # don't forget the last group
        result.append(''.join(current).replace('_', ' '))
    
    return result

examples = {}
def get_examples(feature_idx, sae_layer=31, api_key=None):
    """
    get examples for a given feature from neuronpedia
    """

    # first check if already computed, just return
    if feature_idx in examples:
        return examples[feature_idx]
    
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
    threshold = all_bins[-1]

    all_examples = []
    for i in range(len(acts)):
        contiguous_tokens = get_contiguous_tokens(acts[i]['tokens'], acts[i]['values'], threshold)
        all_examples.extend(contiguous_tokens)

    examples[feature_idx] = all_examples
    return all_examples

async def main():

    results = []
    df = pd.read_csv(args.input_file)
    if args.examples:
        df['examples'] = df['feature_idx'].apply(lambda x: get_examples(x))

    classifier = Classifier(max_concurrents=args.max_concurrents, prompt_file=args.prompt)

    for i in range(0, len(df), args.batch_size):
        
        batch_size = min(args.batch_size, len(df) - i)
        batch = df.iloc[i:i + batch_size]
        batch = batch.to_dict(orient='records')

        annotated = await classifier.process_sentences(batch)
        results.extend(annotated)
        print(f'last description: {annotated[-1]["feature_description"]}')
        break

    df = pd.DataFrame(results)
    if args.examples:
        df.to_csv(args.input_file.split('.')[0] + '-medical-examples.csv', index=False)
    else:
        df.to_csv(args.input_file.split('.')[0] + '-medical.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())