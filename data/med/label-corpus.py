"""
creates a set of documents for training a probe on medical content
   - pull from fineweb and from pre-prepared data mixture
   - break documents into sentences
   - ask claude 4 opus "does this sentence pertain to medicine?"
   - if yes, add to + set
   - if no, add to - set
   - repeat until 10k fineweb documents + 10K other documents in each set
   - save to file
"""

import argparse
import asyncio
import os
import sys
import logging
import json
import random
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

# suppress verbose logging from safetytooling
logging.getLogger('safetytooling').setLevel(logging.WARNING)

# add safety-tooling to path if not already installed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
safety_tooling_path = os.path.join(parent_dir, "safety-tooling")
sys.path.append(safety_tooling_path)

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("downloading NLTK punkt_tab...")
    nltk.download('punkt_tab')

sys.path.append('../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=os.path.join(DATA_PATH, 'docs'))
parser.add_argument("--prompt", type=str, default="sentence_prompt.txt")
parser.add_argument("--annotation_type", type=str, default="sentence")
parser.add_argument("--max_concurrents", type=int, default=20)
parser.add_argument("--target_docs", type=int, default=125000)
args = parser.parse_args()

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

    async def classify_sentence(self, document, sent=''):

        # classify a single document
        async with self.semaphore:

            try:

                # create the prompt
                if args.annotation_type == "sentence":
                    prompt = Prompt(messages = [
                        ChatMessage(content=self.system_prompt, role=MessageRole.system),
                        ChatMessage(content=f'Document: {document}\nSentence: {sent}', role=MessageRole.user)
                    ])
                elif args.annotation_type == "document" or args.annotation_type == "token":
                    prompt = Prompt(messages= [
                        ChatMessage(content=self.system_prompt, role=MessageRole.system),
                        ChatMessage(content=f'Document: {document}', role=MessageRole.user)
                    ])
                
                # make API call
                response = await self.api(
                    model_id="claude-sonnet-4-20250514",
                    prompt=prompt,
                    max_attempts_per_api_call=3,
                    n=1
                )
                
                # parse response
                if args.annotation_type == "token":
                    answer = response[0].completion.strip()
                    try:
                        return [x.strip() for x in answer.split(',')] if answer != "NONE" else []
                    except:
                        print(f"warning: improperly formatted response for document")
                        return []
                else:
                    if response and len(response) > 0:
                        answer = response[0].completion.strip().upper()
                        is_medical = answer.startswith("YES")
                        return is_medical
                    else:
                        print(f"warning: no response for document")
                        return []
                    
            except Exception as e:

                print(f"error classifying document: {e}")
                return [] if args.annotation_type == "token" else False
    
    async def process_sentences(self, sentences, target_docs):

        # process in parallel and classify them
        annotated = []
        num_medical = 0
        num_nonmedical = 0

        # process documents in batches
        batch_size = self.max_concurrents * 2 # process more than concurrent limit to keep queue full
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # skip if we already have enough documents
            if num_medical >= target_docs and num_nonmedical >= target_docs:
                break
            
            print(f"processing batch {i//batch_size + 1}, sentences {i}-{min(i+batch_size, len(sentences))}")
            
            # create tasks for this batch
            tasks = []
            for sentence in batch:
                if args.annotation_type == "sentence":
                    tasks.append(self.classify_sentence(sentence['document'], sentence['text']))
                elif args.annotation_type == "document" or args.annotation_type == "token":
                    tasks.append(self.classify_sentence(sentence['document']))
            
            # process batch concurrently
            results = await tqdm.gather(*tasks, desc="classifying sentences")
            
            if args.annotation_type == "sentence":
                # if the document has any medical, we'll add all of the non-medical
                # to the annotated documents as well to make sure the whole doc is represented
                medical_doc_ids = set()
                for sentence, is_medical in zip(batch, results):
                    if is_medical:
                        medical_doc_ids.add(sentence['doc_id'])
                
                non_medical_doc_ids = set()
                for sentence in batch:
                    if sentence['doc_id'] not in medical_doc_ids:
                        # half the number of pure non-medical documents as medical
                        if len(non_medical_doc_ids) < len(medical_doc_ids) // 3:
                            non_medical_doc_ids.add(sentence['doc_id'])
                
                # collect results
                all_doc_ids = set()
                for sentence, is_medical in zip(batch, results):

                    # if the sentence is in a medical document, and we don't have enough medical sentences, add it
                    # (or if we've already added a sentence from the document)
                    if sentence['doc_id'] in medical_doc_ids and (num_medical < target_docs or sentence['doc_id'] in all_doc_ids):
                        all_doc_ids.add(sentence['doc_id'])

                        if is_medical:
                            annotated.append({
                                'source': sentence.get('source', 'fineweb'),
                                'doc_id': sentence['doc_id'],
                                'sent_id': sentence['sent_id'],
                                'text': sentence['text'],
                                'medical': True,
                            })
                            num_medical += 1
                        
                        else:
                            annotated.append({
                                'source': sentence.get('source', 'fineweb'),
                                'doc_id': sentence['doc_id'],
                                'sent_id': sentence['sent_id'],
                                'text': sentence['text'],
                                'medical': False,
                            })
                            num_nonmedical += 1
                    
                    # if the sentence is not in a medical document, and we don't have enough non-medical sentences, add it
                    # (or if we've already added a sentence from the document)
                    # the main issue is that this will add a lot of non-medical sentences pretty quickly (since fineweb is mostly non-medical)
                    # so we randomly sample to make sure we only have a few "pure" non-medical documents
                    elif sentence['doc_id'] in non_medical_doc_ids and (num_nonmedical < target_docs or sentence['doc_id'] in all_doc_ids):
                        all_doc_ids.add(sentence['doc_id'])
                        annotated.append({
                            'source': sentence.get('source', 'fineweb'),
                            'doc_id': sentence['doc_id'],
                            'sent_id': sentence['sent_id'],
                            'text': sentence['text'],
                            'medical': False,
                        })
                        num_nonmedical += 1
            
            elif args.annotation_type == "document":
                for sentence, is_medical in zip(batch, results):
                    if is_medical and num_medical < target_docs:
                        annotated.append({
                            'source': sentence.get('source', 'fineweb'),
                            'text': sentence['document'],
                            'medical': True,
                        })
                        num_medical += 1
                    elif not is_medical and num_nonmedical < target_docs:
                        annotated.append({
                            'source': sentence.get('source', 'fineweb'),
                            'text': sentence['document'],
                            'medical': False,
                        })
                        num_nonmedical += 1
            
            elif args.annotation_type == "token":
                for sentence, tokens in zip(batch, results):
                    if len(tokens) > 0 and num_medical < target_docs:
                        annotated.append({
                            'source': sentence.get('source', 'fineweb'),
                            'text': sentence['document'],
                            'tokens': tokens,
                            'medical': True,
                        })
                        num_medical += 1
                    elif len(tokens) == 0 and num_nonmedical < target_docs:
                        annotated.append({
                            'source': sentence.get('source', 'fineweb'),
                            'text': sentence['document'],
                            'tokens': tokens,
                            'medical': False,
                        })
                        num_nonmedical += 1
            # show progress
            print(f"progress: {num_medical} medical docs, {num_nonmedical} non-medical docs")
            
            # # break if we have enough documents
            # if num_medical >= target_docs and num_nonmedical >= target_docs:
            #     break
        
        return annotated, num_medical, num_nonmedical

def append_results(batch_results, output_dir):
    """append new batch results to existing files"""
    os.makedirs(output_dir, exist_ok=True)
    
    if args.annotation_type == "sentence":
        file = os.path.join(output_dir, "sentences-new.jsonl")
    elif args.annotation_type == "document":
        file = os.path.join(output_dir, "documents-new.jsonl")
    elif args.annotation_type == "token":
        file = os.path.join(output_dir, "tokens-new.jsonl")
    with open(file, 'a', encoding='utf-8') as f:
        for doc in batch_results:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

async def main():

    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", streaming = True)
    gold = pd.read_json('gold.jsonl', orient='records', lines=True)
    gold = Dataset.from_pandas(gold)
    gold = gold.shuffle(seed=42)

    batch_size = min(240, args.target_docs // 20) # reasonable batch size
    classifier = Classifier(max_concurrents=args.max_concurrents, prompt_file=args.prompt)

    total_processed = 0
    offset = 0

    total_medical = 0
    total_nonmedical = 0

    # clear output files at the beginning
    os.makedirs(args.data_path, exist_ok=True)
    if args.annotation_type == "sentence":
        file = os.path.join(args.data_path, "sentences-new.jsonl")
    elif args.annotation_type == "document":
        file = os.path.join(args.data_path, "documents-new.jsonl")
    elif args.annotation_type == "token":
        file = os.path.join(args.data_path, "tokens-new.jsonl")
    
    # clear existing files
    with open(file, 'w', encoding='utf-8') as f:
        pass

    while (total_medical < (3 * args.target_docs // 2) or total_nonmedical < (3 * args.target_docs // 2)) and args.annotation_type != 'document':

        batch = gold.skip(offset).take(batch_size)

        if args.annotation_type == "sentence":
            sents = [sent_tokenize(ex) for ex in batch['text']]
            sentences = []

            for i, row in enumerate(batch):
                sentences.extend([{
                    'source': row['source'],
                    'doc_id': hash(row['text']),
                    'sent_id': j,
                    'document': row['text'],
                    'text': sent,
                } for j, sent in enumerate(sents[i])])
        
        elif args.annotation_type == "document" or args.annotation_type == "token":
            sentences = [{
                'source': row['source'],
                'document': row['text']
            } for row in batch]
        
        results, num_medical, num_nonmedical = await classifier.process_sentences(sentences, 3 * args.target_docs // 2)

        medical_needed    = max(3 * args.target_docs // 2 - total_medical, 0)
        nonmedical_needed = max(3 * args.target_docs // 2 - total_nonmedical, 0)

        total_medical += min(num_medical, medical_needed)
        total_nonmedical += min(num_nonmedical, nonmedical_needed)
        append_results(results, args.data_path)

        offset += batch_size
        total_processed += batch_size

        print(f"processed {total_processed} | {total_medical} medical | {total_nonmedical} nonmedical")

    offset = 0
    print(f'processing fineweb...')
    while ((total_medical < 2 * args.target_docs or total_nonmedical < 2 * args.target_docs) and args.annotation_type != 'document') or (args.annotation_type == 'document' and (total_medical < args.target_docs // 2 or total_nonmedical < args.target_docs // 2)):

        batch = fineweb['train'].skip(offset).take(batch_size)
        
        if args.annotation_type == "sentence":
            sents = [sent_tokenize(ex['text']) for ex in batch]
            sentences = []

            for i, row in enumerate(batch):
                sentences.extend([{
                    'source': 'fineweb',
                    'doc_id': hash(row['text']),
                    'sent_id': j,
                    'document': row['text'],
                    'text': sent,
                } for j, sent in enumerate(sents[i])])
        
        elif args.annotation_type == "document" or args.annotation_type == "token":
            
            # some fineweb parquets are malformed
            try:
                sentences = [{
                    'source': 'fineweb',
                    'document': row['text']
                } for row in batch]
            
            except:
                print(f"error processing batch {offset}")
                continue
        
        results, num_medical, num_nonmedical = await classifier.process_sentences(sentences, args.target_docs // 2)

        if args.annotation_type == 'document':
            medical_needed = max(args.target_docs // 2 - total_medical, 0)
            nonmedical_needed = max(args.target_docs // 2 - total_nonmedical, 0)
        else:
            medical_needed = max(2 * args.target_docs - total_medical, 0)
            nonmedical_needed = max(2 * args.target_docs - total_nonmedical, 0)

        total_medical += min(num_medical, medical_needed)
        total_nonmedical += min(num_nonmedical, nonmedical_needed)

        append_results(results, args.data_path)

        offset += batch_size
        total_processed += batch_size

        print(f"processed {total_processed} | {total_medical} medical | {total_nonmedical} nonmedical")
    
    print(f"done")

if __name__ == "__main__":
    asyncio.run(main())