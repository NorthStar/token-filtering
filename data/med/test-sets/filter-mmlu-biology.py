"""
Filter MMLU biology questions to remove medical content using Claude API.
"""

import argparse
import asyncio
import os
import sys
import logging
import json
from pathlib import Path
from tqdm.asyncio import tqdm
from datasets import load_dataset, concatenate_datasets

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

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "../../../evals"))
parser.add_argument("--prompt", type=str, default=os.path.join(os.path.dirname(__file__), "..", "mmlu_prompt.txt"))
parser.add_argument("--max_concurrents", type=int, default=20)
args = parser.parse_args()


def format_question_for_classifier(question):
    """Format MMLU question for the medical classifier."""
    text = f"{question['question']}\n\n"
    for i, choice in enumerate(question['choices']):
        text += f"{chr(65 + i)}. {choice}\n"
    return text


def format_question_for_output(question):
    """Format MMLU question for the output jsonl (matching eval format)."""
    formatted_choices = ""
    for i, choice in enumerate(question['choices']):
        formatted_choices += f"Choice: {choice} = {chr(65 + i)}\n"
    
    input_text = f"""Question: {question['question']}\n\nChoices:\n{formatted_choices}\nAnswer:"""
    
    output = chr(65 + question['answer'])  # 0 -> A, 1 -> B, etc.
    return input_text, output


class Classifier:
    def __init__(self, max_concurrents=5, prompt_file=None):
        """Initialize the classifier with safety-tooling API."""
        self.max_concurrents = max_concurrents
        self.semaphore = asyncio.Semaphore(max_concurrents)
        
        # setup environment for API keys
        utils.setup_environment()
        
        # initialize the inference API
        self.api = InferenceAPI(
            cache_dir=Path(".cache"),
            anthropic_num_threads=max_concurrents
        )
        
        # load system prompt from file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
        print(f"loaded prompt from {prompt_file}")

    async def classify_question(self, question_text):
        """Classify a single question as medical or not."""
        async with self.semaphore:
            try:
                prompt = Prompt(messages=[
                    ChatMessage(content=self.system_prompt, role=MessageRole.system),
                    ChatMessage(content=f'Document: {question_text}', role=MessageRole.user)
                ])
                
                response = await self.api(
                    model_id="claude-sonnet-4-20250514",
                    prompt=prompt,
                    max_attempts_per_api_call=3,
                    n=1
                )
                
                if response and len(response) > 0:
                    answer = response[0].completion.strip().upper()
                    return answer.startswith("YES")
                else:
                    print(f"warning: no response for question")
                    return None
                    
            except Exception as e:
                print(f"error classifying question: {e}")
                return None

    async def filter_questions(self, questions):
        """Process questions and filter out medical ones."""
        non_medical = []
        batch_size = self.max_concurrents * 2
        
        num_medical = 0
        num_non_medical = 0
        num_errors = 0
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            print(f"processing batch {i//batch_size + 1}, questions {i}-{min(i+batch_size, len(questions))}")
            
            # Format questions for classifier
            question_texts = [format_question_for_classifier(q) for q in batch]
            
            tasks = [self.classify_question(text) for text in question_texts]
            results = await tqdm.gather(*tasks, desc="classifying questions")
            
            for question, is_medical in zip(batch, results):
                if is_medical is None:
                    num_errors += 1
                    continue
                    
                if is_medical:
                    num_medical += 1
                else:
                    non_medical.append(question)
                    num_non_medical += 1
            
            print(f"progress: {num_non_medical} non-medical, {num_medical} medical, {num_errors} errors")
        
        return non_medical


async def main():
    # Load MMLU biology subsets
    print("Loading MMLU biology datasets...")

    datasets = []

    mmlu = load_dataset("cais/mmlu", 'all', split='test')
    for subject in ['college_biology', 'high_school_biology', 'college_chemistry', 'high_school_chemistry']:

        ds = mmlu.filter(lambda x: x['subject'] == subject)
        datasets.append(ds)

    mmlu_val = load_dataset("cais/mmlu", 'all', split='validation')
    for subject in ['college_biology', 'high_school_biology', 'college_chemistry', 'high_school_chemistry']:

        ds = mmlu_val.filter(lambda x: x['subject'] == subject)
        datasets.append(ds)
    
    all_mmlu = concatenate_datasets(datasets)
    questions = list(all_mmlu)
    
    print(f"loaded {len(questions)} MMLU questions total")
    
    classifier = Classifier(max_concurrents=args.max_concurrents, prompt_file=args.prompt)
    
    non_medical = await classifier.filter_questions(questions)
    
    # Write filtered questions to jsonl
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "mmlu_biology.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in non_medical:
            input_text, output = format_question_for_output(question)
            line = {'input': input_text, 'output': output}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    print(f"saved {len(non_medical)} non-medical questions to {output_file}")
    print("done")


if __name__ == "__main__":
    asyncio.run(main())

