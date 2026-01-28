"""
Sample questions from multimedqa-graded and combine responses from all four model types.
"""

import os
import sys
import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('num_questions', type=int, help='number of questions to sample')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)

# Paths
input_dir = os.path.join(os.path.dirname(__file__), 'results/multimedqa-graded')
output_path = os.path.join(os.path.dirname(__file__), 'results/sampled-multimedqa-responses.jsonl')

# Model types to load
model_types = ['nomask', 'mask', 'remove', 'document']

# Load all responses, indexed by question
responses_by_question = {}

for model_type in model_types:
    filepath = os.path.join(input_dir, f'{model_type}.jsonl')

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            question = item['question']

            if question not in responses_by_question:
                responses_by_question[question] = {}

            responses_by_question[question][model_type] = item['answer']

# Filter to questions that have responses from all four model types
complete_questions = [
    q for q, responses in responses_by_question.items()
    if all(mt in responses for mt in model_types)
]

print(f"Total questions with all four model responses: {len(complete_questions)}")

if args.num_questions > len(complete_questions):
    print(f"Warning: Requested {args.num_questions} questions but only {len(complete_questions)} available")
    args.num_questions = len(complete_questions)

# Sample questions
sampled_questions = random.sample(complete_questions, args.num_questions)

print(f"Sampled {len(sampled_questions)} questions")

# Write output
with open(output_path, 'w') as f:
    for question in sampled_questions:
        output_obj = {
            'question': question,
            'nomask': responses_by_question[question]['nomask'],
            'mask': responses_by_question[question]['mask'],
            'remove': responses_by_question[question]['remove'],
            'document': responses_by_question[question]['document']
        }
        f.write(json.dumps(output_obj) + '\n')

print(f"Saved {len(sampled_questions)} sampled responses to {output_path}")
