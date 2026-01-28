"""
Evaluate instruct and base models on MMLU and medical QA benchmarks.

This script:
1. Evaluates instruct-tuned models on multiple datasets (MMLU, MedQA, etc.)
   - Computes both accuracy and negative log likelihood (NLL) in a single forward pass
   - Saves results to mmlu.csv and mmlu-nll.csv
   
2. Evaluates base models using cloze-style evaluation
   - Saves results to mmlu-cloze.csv
   
3. Generates 6 plots comparing different filtering methods:
   - Max params instruct models (accuracy)
   - Max params base models (accuracy)
   - All params base models (accuracy)
   - All params instruct models (accuracy)
   - Max params instruct models (NLL - lower is better)
   - All params instruct models (NLL - lower is better)

Usage:
    python mmlu.py [--rerun_instruct] [--rerun_cloze] [--max_params_only]
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
import tiktoken
from datasets import load_dataset, concatenate_datasets
from colors import MASK_COLORS, MASK_LABELS, get_mask_color_list, THEME_COLORS
import sys
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use Helvetica Neue for mathtext
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica Neue'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH
from eval_utils import (
    predict_mcq, predict_mcq_cloze, load_model, build_fewshot_prompt,
    format_mmlu_question, format_medmcqa_question, format_medqa_question,
    format_pubmedqa_question, format_headqa_question, format_medconceptsqa_question,
    format_jsonl_question, load_hf_dataset, load_jsonl_dataset
)

def calculate_se_bounds(predictions):
    """
    Calculate ±1 standard error bounds for accuracy
    
    Args:
        predictions: list of boolean values (True for correct, False for incorrect)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if len(predictions) == 0:
        return 0.0, 0.0
    
    predictions = np.array(predictions, dtype=float)
    mean_accuracy = np.mean(predictions)
    std_accuracy = np.std(predictions, ddof=1)
    n_samples = len(predictions)
    se_accuracy = std_accuracy / np.sqrt(n_samples)
    
    lower_bound = mean_accuracy - se_accuracy
    upper_bound = mean_accuracy + se_accuracy
    
    return lower_bound, upper_bound

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'instruct'), help='path to models')
parser.add_argument('--base_model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to base models')
parser.add_argument('--n_fewshot', type=int, default=0, help='number of few-shot examples')
parser.add_argument('--max_samples', type=int, default=10000, help='maximum number of test samples per dataset')
parser.add_argument('--max_params_only', action='store_true', help='only evaluate largest model size')
parser.add_argument('--rerun_instruct', action='store_true', help='rerun instruct model evaluation (produces mmlu.csv and mmlu-nll.csv)')
parser.add_argument('--rerun_cloze', action='store_true', help='rerun base model cloze evaluation (produces mmlu-cloze.csv)')
args = parser.parse_args()

def get_max_params_from_dir(model_dir):
    """Get the maximum parameter count from models in a directory"""
    max_params = 0
    for model_file in os.listdir(model_dir):
        if not model_file.endswith('.pt') or 'old' in model_file:
            continue
        try:
            num_str = model_file.split('-')[1]
            if 'M' in num_str:
                num_params = int(num_str.split('M')[0]) * 1e6
                max_params = max(max_params, num_params)
        except (IndexError, ValueError):
            continue
    return max_params

def evaluate_mmlu(model, dset, fewshot, enc, device, cloze=False):
    """
    Evaluate a model on a dataset, computing both accuracy and log likelihoods.
    
    Args:
        model: GPT model to evaluate
        dset: dataset dict with 'input' and 'output' keys
        fewshot: few-shot examples dict
        enc: tokenizer
        device: compute device
        cloze: whether to use cloze evaluation (base models only)
    
    Returns:
        tuple: (correct, total, correct_truncated, total_truncated, 
                predictions, truncated_mask, log_likelihoods)
    """
    fewshot_prompt = build_fewshot_prompt(fewshot)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else torch.no_grad()

    correct = 0
    total = len(dset['input'])
    correct_truncated = 0
    total_truncated = 0
    
    predictions = []
    truncated_mask = []
    nlls = [] if not cloze else None  # only compute NLL for instruct models

    for i in range(0, len(dset['input'])):
        
        if cloze:
            try:
                predicted_letter, truncated = predict_mcq_cloze(model, dset['input'][i], fewshot_prompt, enc, device, ctx)
            except Exception as e:
                predicted_letter = 'X'
                truncated = True
        else:
            predicted_letter, truncated, answer_nll = predict_mcq(model, dset['input'][i], fewshot_prompt, enc, device, ctx, return_log_probs=True)
            # Store NLL of predicted answer (lower = more confident)
            nlls.append(answer_nll.get(predicted_letter, 0.0))

        is_correct = predicted_letter == dset['output'][i]
        
        predictions.append(is_correct)
        truncated_mask.append(truncated)
        
        if is_correct:
            correct += 1
        if truncated:
            total_truncated += 1
        if is_correct and truncated:
            correct_truncated += 1
    
    return correct, total, correct_truncated, total_truncated, predictions, truncated_mask, nlls

def evaluate_and_record(model, dset, fewshot, enc, device, subject_name, num_params, mask, results, results_nll, cloze=False):
    """
    Evaluate a dataset and record results to both accuracy and NLL dicts.
    
    Args:
        model: GPT model
        dset, fewshot: dataset and few-shot examples
        enc, device: tokenizer and compute device
        subject_name: name of the dataset/subject
        num_params: model parameter count
        mask: filtering method name
        results: dict to store accuracy metrics
        results_nll: dict to store negative log likelihood metrics (None for cloze)
        cloze: whether using cloze evaluation
    """
    print(f'evaluating {subject_name}')
    correct, total, correct_truncated, total_truncated, predictions, truncated_mask, nlls = evaluate_mmlu(
        model, dset, fewshot, enc, device, cloze=cloze
    )
    
    # Filter out truncated samples
    non_truncated_predictions = [pred for pred, trunc in zip(predictions, truncated_mask) if not trunc]
    
    if len(non_truncated_predictions) > 0:
        acc = np.mean(non_truncated_predictions)
        se_lower, se_upper = calculate_se_bounds(non_truncated_predictions)
    else:
        acc = 0
        se_lower, se_upper = 0, 0
    
    # Compute NLL stats only for non-cloze evaluations
    if nlls is not None:
        non_truncated_nll = [nll for nll, trunc in zip(nlls, truncated_mask) if not trunc]
        if len(non_truncated_nll) > 0:
            avg_nll = np.mean(non_truncated_nll)
            nll_std = np.std(non_truncated_nll)
        else:
            avg_nll = 0
            nll_std = 0
        print('accuracy:', acc, f'avg NLL: {avg_nll:.4f}', f'({total_truncated/total:.2f} truncated)', f'±1SE: [{se_lower:.3f}, {se_upper:.3f}]')
    else:
        print('accuracy:', acc, f'({total_truncated/total:.2f} truncated)', f'±1SE: [{se_lower:.3f}, {se_upper:.3f}]')

    # Record accuracy results
    results['subject'].append(subject_name)
    results['params'].append(num_params)
    results['mask'].append(mask)
    results['accuracy'].append(acc)
    results['se_lower'].append(se_lower)
    results['se_upper'].append(se_upper)
    
    # Record NLL results (only for non-cloze)
    if not cloze and results_nll is not None:
        results_nll['subject'].append(subject_name)
        results_nll['params'].append(num_params)
        results_nll['mask'].append(mask)
        results_nll['accuracy'].append(acc)
        results_nll['avg_nll'].append(avg_nll)
        results_nll['nll_std'].append(nll_std)
        results_nll['se_lower'].append(se_lower)
        results_nll['se_upper'].append(se_upper)

def load_all_datasets():
    """
    Load and prepare all evaluation datasets from HuggingFace and local JSONL files.
    
    Returns:
        tuple: (mmlu_data dict, mmlu_subjects dict, datasets dict)
    """
    
    # MMLU subjects (mmlu biology loaded separately from pre-filtered jsonl)
    mmlu_subjects = {
        'mmlu stem': ['conceptual_physics', 'high_school_physics', 'college_physics', 
                      'high_school_computer_science', 'college_computer_science', 
                      'electrical_engineering', 'machine_learning', 'astronomy', 'computer_security'],
        'mmlu medicine': ['college_medicine', 'professional_medicine', 'medical_genetics',
                      'clinical_knowledge', 'anatomy', 'virology'],
        'mmlu non-stem': ['high_school_government_and_politics', 'high_school_geography', 
                          'international_law', 'jurisprudence', 'management', 'marketing', 
                          'philosophy', 'prehistory', 'public_relations', 'sociology', 
                          'us_foreign_policy', 'world_religions'],
    }

    # Load HuggingFace datasets once
    print("Loading datasets from HuggingFace...")
    mmlu_val = load_dataset("cais/mmlu", 'all', split='validation')
    mmlu_test = load_dataset("cais/mmlu", 'all', split='test')
    
    medqa_val = load_dataset("GBaker/MedQA-USMLE-4-options", split='test')
    medqa_test = load_dataset("GBaker/MedQA-USMLE-4-options", split='train')
    
    medmcqa_val = load_dataset("openlifescienceai/medmcqa", split='validation')
    medmcqa_test = load_dataset("openlifescienceai/medmcqa", split='train')
    medmcqa_ignore = ['Biochemistry', 'Microbiology', 'Psychiatry', 'Social & Preventive Medicine']
    medmcqa_test = medmcqa_test.filter(lambda x: x['subject_name'] not in medmcqa_ignore)
    
    pubmedqa_val = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train')
    pubmedqa_test = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train')
    
    # headqa_val = load_dataset("EleutherAI/headqa", 'en', split='validation')
    # headqa_train = load_dataset("EleutherAI/headqa", 'en', split='train')
    # headqa_test = load_dataset("EleutherAI/headqa", 'en', split='test')
    # headqa_val = headqa_val.filter(lambda x: x['category'] not in ['biology', 'chemistry', 'psychology'])
    # headqa_train = headqa_train.filter(lambda x: x['category'] not in ['biology', 'chemistry', 'psychology'])
    # headqa_test = headqa_test.filter(lambda x: x['category'] not in ['biology', 'chemistry', 'psychology'])
    # headqa_eval = concatenate_datasets([headqa_train, headqa_test])
    
    # Load MMLU subjects
    mmlu_data = {}
    for subject_category in mmlu_subjects:
        for subject in mmlu_subjects[subject_category]:
            val = mmlu_val.filter(lambda x: x['subject'] == subject)
            test = mmlu_test.filter(lambda x: x['subject'] == subject)
            dset, fewshot = load_hf_dataset(val, test, format_mmlu_question, args.n_fewshot, args.max_samples)
            mmlu_data[subject] = (dset, fewshot)
    
    # Configure all datasets for evaluation
    datasets = {}
    
    # Load medical datasets
    datasets['medqa-usmle'] = load_hf_dataset(medqa_val, medqa_test, format_medqa_question, 
                                               args.n_fewshot, args.max_samples)
    datasets['medmcqa'] = load_hf_dataset(medmcqa_val, medmcqa_test, format_medmcqa_question, 
                                          args.n_fewshot, args.max_samples)
    datasets['pubmedqa'] = load_hf_dataset(pubmedqa_val, pubmedqa_test, format_pubmedqa_question, 
                                           args.n_fewshot, args.max_samples)
    # datasets['headqa'] = load_hf_dataset(headqa_val, headqa_eval, format_headqa_question, 
    #                                      args.n_fewshot, args.max_samples)
    
    # Load local JSONL datasets
    # datasets['medqa-mcmle'] = load_jsonl_dataset('../data/evals/medqa-mcmle.jsonl', format_jsonl_question, 
    #                                              args.n_fewshot, args.max_samples, 
    #                                              split_field='split', val_splits=['dev'], 
    #                                              test_splits=['train', 'test'])
    
    # Load pre-filtered MMLU biology (medical questions removed)
    datasets['mmlu biology'] = load_jsonl_dataset('../evals/mmlu_biology.jsonl', format_jsonl_question,
                                                   args.n_fewshot, args.max_samples)
    
    return mmlu_data, mmlu_subjects, datasets

def evaluate_models(model_path, datasets, mmlu_data, mmlu_subjects, enc, cloze=False):
    """
    Evaluate all models in a directory on all datasets.
    
    For instruct models (cloze=False): collects both accuracy and NLL in one pass.
    For base models (cloze=True): collects only accuracy.
    
    Args:
        model_path: directory containing model .pt files
        datasets: dict of test datasets
        mmlu_data: dict of MMLU subject data
        mmlu_subjects: dict mapping subject categories to individual subjects
        enc: tokenizer
        cloze: whether to use cloze evaluation
    
    Returns:
        tuple: (results dict, results_nll dict or None, max_params)
    """
    results = {
        'subject': [],
        'params': [],
        'mask': [],
        'accuracy': [],
        'se_lower': [],
        'se_upper': []
    }
    
    results_nll = None if cloze else {
        'subject': [],
        'params': [],
        'mask': [],
        'accuracy': [],
        'avg_nll': [],
        'nll_std': [],
        'se_lower': [],
        'se_upper': []
    }
    
    max_params = 0
    target_params = None
    if args.max_params_only:
        target_params = get_max_params_from_dir(model_path)
        print(f'Only evaluating models with {int(target_params/1e6)}M parameters')

    for model_file in sorted(os.listdir(model_path)):
        if 'old' in model_file or not model_file.endswith('.pt'):
            continue
        
        # Parse model params
        try:
            num_str = model_file.split('-')[1]
            if 'M' in num_str:
                num_params = int(num_str.split('M')[0]) * 1e6
            else:
                num_params = int(num_str.split('.')[0]) * 1e6
        except (IndexError, ValueError):
            continue
        
        # Filter by max params if requested
        if args.max_params_only and num_params != target_params:
            continue
        
        model = load_model(os.path.join(model_path, model_file), args.device)
        mask = model_file.split('-')[0]
        
        if num_params > max_params:
            max_params = num_params
        
        # Evaluate MMLU subject categories
        for subject_category in mmlu_subjects:
            print(f'evaluating {subject_category}')
            all_predictions = []
            all_truncated = []
            all_nlls = [] if not cloze else None

            for subject in mmlu_subjects[subject_category]:
                dset, fewshot = mmlu_data[subject]
                correct, total, correct_truncated, total_truncated, predictions, truncated_mask, nlls = \
                    evaluate_mmlu(model, dset, fewshot, enc, args.device, cloze=cloze)

                if total_truncated > 0.5 * total:
                    print(f'skipping {subject} as {total_truncated / total:.2f} docs truncated')
                    continue

                all_predictions.extend(predictions)
                all_truncated.extend(truncated_mask)
                if nlls is not None:
                    all_nlls.extend(nlls)
            
            # Calculate metrics with SE bounds
            non_truncated_predictions = [pred for pred, trunc in zip(all_predictions, all_truncated) if not trunc]
            
            if len(non_truncated_predictions) > 0:
                acc = np.mean(non_truncated_predictions)
                se_lower, se_upper = calculate_se_bounds(non_truncated_predictions)
                total_truncated = sum(all_truncated)
                total_items = len(all_predictions)
            else:
                acc = 0
                se_lower, se_upper = 0, 0
                total_truncated = sum(all_truncated)
                total_items = len(all_predictions)
            
            # Calculate NLL stats only for non-cloze
            if all_nlls is not None:
                non_truncated_nll = [nll for nll, trunc in zip(all_nlls, all_truncated) if not trunc]
                if len(non_truncated_nll) > 0:
                    avg_nll = np.mean(non_truncated_nll)
                    nll_std = np.std(non_truncated_nll)
                else:
                    avg_nll = 0
                    nll_std = 0
                print('accuracy:', acc, f'avg NLL: {avg_nll:.4f}', f'({total_truncated/total_items:.2f} truncated)', f'±1SE: [{se_lower:.3f}, {se_upper:.3f}]')
            else:
                print('accuracy:', acc, f'({total_truncated/total_items:.2f} truncated)', f'±1SE: [{se_lower:.3f}, {se_upper:.3f}]')

            results['subject'].append(subject_category)
            results['params'].append(num_params)
            results['mask'].append(mask)
            results['accuracy'].append(acc)
            results['se_lower'].append(se_lower)
            results['se_upper'].append(se_upper)
            
            # Also record NLL results (only for non-cloze)
            if not cloze:
                results_nll['subject'].append(subject_category)
                results_nll['params'].append(num_params)
                results_nll['mask'].append(mask)
                results_nll['accuracy'].append(acc)
                results_nll['avg_nll'].append(avg_nll)
                results_nll['nll_std'].append(nll_std)
                results_nll['se_lower'].append(se_lower)
                results_nll['se_upper'].append(se_upper)
        
        # Evaluate all other datasets
        for subject_name, (dset, fewshot) in datasets.items():
            evaluate_and_record(model, dset, fewshot, enc, args.device, 
                              subject_name, num_params, mask, results, results_nll, cloze=cloze)
    
    return results, results_nll, max_params

# Main evaluation
results_exist = os.path.exists('results/mmlu.csv') and os.path.exists('results/mmlu-nll.csv')
cloze_exists = os.path.exists('results/mmlu-cloze.csv')

if not results_exist or not cloze_exists or args.rerun_instruct or args.rerun_cloze:
    
    cl100k_base = tiktoken.get_encoding('cl100k_base')
    enc = tiktoken.Encoding(
        name="cl100k_mask",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|mask|>": 100277
        }
    )

    mmlu_data, mmlu_subjects, datasets = load_all_datasets()
    
    # Evaluate instruct models (collects both accuracy and NLL in one pass)
    if not results_exist or args.rerun_instruct:
        print("\n=== Evaluating instruct models ===")
        print("(Collecting accuracy and negative log likelihood in a single pass)")
        results, results_nll, max_params = evaluate_models(args.model_path, datasets, mmlu_data, 
                                                           mmlu_subjects, enc, cloze=False)
        
        df = pd.DataFrame(results)
        df.to_csv('results/mmlu.csv', index=False)
        print(f"Saved results/mmlu.csv")
        
        if results_nll is not None:
            df_nll = pd.DataFrame(results_nll)
            df_nll.to_csv('results/mmlu-nll.csv', index=False)
            print(f"Saved results/mmlu-nll.csv")
    
    # Evaluate base models with cloze
    if not cloze_exists or args.rerun_cloze:
        print("\n=== Evaluating base models (cloze) ===")
        results_cloze, _, max_params_cloze = evaluate_models(args.base_model_path, datasets, 
                                                             mmlu_data, mmlu_subjects, enc, cloze=True)
        df_cloze = pd.DataFrame(results_cloze)
        df_cloze.to_csv('results/mmlu-cloze.csv', index=False)
        print(f"Saved results/mmlu-cloze.csv")

# ============================================================================
# Load results and create plots
# ============================================================================
df = pd.read_csv('results/mmlu.csv')
df_cloze = pd.read_csv('results/mmlu-cloze.csv')
df_nll = pd.read_csv('results/mmlu-nll.csv')

# Filter out 'collapse' mask type
df = df[df['mask'] != 'collapse']
df_cloze = df_cloze[df_cloze['mask'] != 'collapse']
df_nll = df_nll[df_nll['mask'] != 'collapse']

max_params = df['params'].max()

# ============================================================================
# Plot configuration
# ============================================================================
breaks = sorted(df['params'].unique())
labels = [f'{int(p / 1e6)}M' for p in breaks]

x_labels_full = {
    # 'pubmedqa': 'PubMedQA',
    'medmcqa': 'MedMCQA',
    'medqa-usmle': 'MedQA-USMLE',
    # 'medqa-mcmle': 'MedQA-MCMLE',
    # 'headqa': 'HeadQA',
    'mmlu medicine': 'MMLU Medicine',
    'mmlu biology': 'MMLU Bio',
    'mmlu stem': 'MMLU STEM',
    'mmlu non-stem': 'MMLU Non-STEM'
}

# For plotting, exclude pubmedqa
x_labels = {k: v for k, v in x_labels_full.items() if k != 'pubmedqa'}

# Subject-specific chance levels for baseline
chance_levels = {
    # 'pubmedqa': 0.33,
    'medmcqa': 0.25,
    'medqa-usmle': 0.25,
    # 'medqa-mcmle': 0.25,
    # 'headqa': 0.20,
    'mmlu medicine': 0.25,
    'mmlu biology': 0.25,
    'mmlu stem': 0.25,
    'mmlu non-stem': 0.25
}

# Use centralized mask labels from colors.py
legend_labels = {m: MASK_LABELS[m] for m in ['nomask', 'document', 'mask', 'remove', 'unlearn']}

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Prepare colors - use centralized mask colors
mask_order = ['nomask', 'document', 'mask', 'remove']
hex_colors = get_mask_color_list(mask_order)

# ============================================================================
# Plot 1: Max params instruct models (accuracy) - faceted by medical/non-medical
# ============================================================================
df = df[df['subject'].isin(x_labels_full.keys())]
df_max_params = df[df['params'] == max_params].copy()

# Define medical vs non-medical subjects
medical_subjects = ['medmcqa', 'medqa-usmle', 'mmlu medicine']
non_medical_subjects = ['mmlu biology', 'mmlu stem', 'mmlu non-stem']

# Filter to only valid masks and drop any NaN
df_max_params = df_max_params[df_max_params['mask'].isin(mask_order)].copy()
df_max_params = df_max_params.dropna(subset=['mask', 'subject', 'accuracy'])

# Convert accuracy to percentage
df_max_params['accuracy'] = df_max_params['accuracy'] * 100

# Add facet column
df_max_params['facet'] = df_max_params['subject'].apply(
    lambda x: 'forget' if x in medical_subjects else 'retain'
)

# Order subjects within each facet
medical_order = [x_labels[s] for s in medical_subjects]
non_medical_order = [x_labels[s] for s in non_medical_subjects]

df_max_params['subject'] = df_max_params['subject'].replace(x_labels)
df_max_params['subject'] = pd.Categorical(
    df_max_params['subject'], 
    categories=medical_order + non_medical_order,
    ordered=True
)
df_max_params['mask'] = df_max_params['mask'].replace(legend_labels)
df_max_params['mask'] = pd.Categorical(
    df_max_params['mask'], 
    categories=[legend_labels[m] for m in ['nomask', 'document', 'mask', 'remove']],
    ordered=True
)
df_max_params['facet'] = pd.Categorical(
    df_max_params['facet'],
    categories=['forget', 'retain'],
    ordered=True
)

# Create annotation data for facet labels (positioned inside top-left of each facet)
# Use global max for consistent positioning across shared y-axis
max_acc = df_max_params['accuracy'].max()
facet_labels = pd.DataFrame({
    'facet': pd.Categorical(['forget', 'retain'], categories=['forget', 'retain'], ordered=True),
    'label': [r'Forget ($\downarrow$)', r'Retain ($\uparrow$)'],
    'subject': [medical_order[0], non_medical_order[0]],  # first subject in each facet
    'y': [max_acc * 0.93, max_acc * 0.93]
})

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_max_params, aes(x='subject', y='accuracy', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + geom_hline(yintercept=25, color=line_color, size=0.5)
        + geom_text(data=facet_labels, mapping=aes(x='subject', y='y', label='label'),
                   ha='left', va='center', size=9, family='Helvetica Neue',
                   fontweight='bold', color=text_color,
                   nudge_x=-0.45, inherit_aes=False)
        + facet_wrap('~facet', scales='free_x')
        + scale_y_continuous(name='Accuracy (%)')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=hex_colors, labels=list(legend_labels.values()))
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(7, 2.625),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            strip_text=element_blank(),
            strip_background=element_blank(),
            legend_position='top',
            legend_key_width=9,
            legend_direction='horizontal',
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color)
        )
)

p.save('plots/mmlu-max-params.png', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params.svg', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params.pdf', dpi=300, width=7, height=2.625)
print("plot saved as 'plots/mmlu-max-params.png'")

# ============================================================================
# Plot 1b: Max params instruct models (normalized accuracy)
# y = (mask_accuracy - 0.25) / (nomask_accuracy - 0.25)
# ============================================================================
# Work with the original df_max_params before label replacement
df_norm = df[df['params'] == max_params].copy()

# Get nomask accuracy for each subject
nomask_acc = df_norm[df_norm['mask'] == 'nomask'].set_index('subject')['accuracy'].to_dict()

# Filter to only document, mask, remove (excluding nomask and collapse)
df_norm = df_norm[df_norm['mask'].isin(['document', 'mask', 'remove'])].copy()

# Calculate normalized accuracy
df_norm['nomask_accuracy'] = df_norm['subject'].map(nomask_acc)
df_norm['normalized_accuracy'] = (df_norm['accuracy'] - 0.25) / (df_norm['nomask_accuracy'] - 0.25) * 100

# Calculate normalized SE bounds (scale by the same denominator)
df_norm['se_lower_norm'] = (df_norm['se_lower'] - 0.25) / (df_norm['nomask_accuracy'] - 0.25) * 100
df_norm['se_upper_norm'] = (df_norm['se_upper'] - 0.25) / (df_norm['nomask_accuracy'] - 0.25) * 100

# Add facet column
df_norm['facet'] = df_norm['subject'].apply(
    lambda x: 'forget' if x in medical_subjects else 'retain'
)

# Set up categorical ordering
df_norm['subject'] = df_norm['subject'].replace(x_labels)
df_norm['subject'] = pd.Categorical(
    df_norm['subject'],
    categories=medical_order + non_medical_order,
    ordered=True
)
df_norm['mask'] = df_norm['mask'].replace(legend_labels)
df_norm['mask'] = pd.Categorical(
    df_norm['mask'],
    categories=[legend_labels[m] for m in ['document', 'mask', 'remove']],
    ordered=True
)
df_norm['facet'] = pd.Categorical(
    df_norm['facet'],
    categories=['forget', 'retain'],
    ordered=True
)

# Filter colors to only the 3 mask types we're using (indices 1, 2, 3 from original)
norm_hex_colors = [hex_colors[1], hex_colors[2], hex_colors[3]]
norm_legend_labels = [legend_labels['document'], legend_labels['mask'], legend_labels['remove']]

# Create annotation data for facet labels
max_norm_acc = df_norm['normalized_accuracy'].max()
facet_labels_norm = pd.DataFrame({
    'facet': pd.Categorical(['forget', 'retain'], categories=['forget', 'retain'], ordered=True),
    'label': [r'Forget ($\downarrow$)', r'Retain ($\uparrow$)'],
    'subject': [medical_order[0], non_medical_order[0]],
    'y': [max_norm_acc * 0.93, max_norm_acc * 0.93]
})

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_norm, aes(x='subject', y='normalized_accuracy', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + geom_hline(yintercept=100, color=line_color, size=0.5)
        + geom_text(data=facet_labels_norm, mapping=aes(x='subject', y='y', label='label'),
                   ha='left', va='center', size=9, family='Helvetica Neue',
                   fontweight='bold', color=text_color,
                   nudge_x=-0.45, inherit_aes=False)
        + facet_wrap('~facet', scales='free_x')
        + scale_y_continuous(name='Normalized Accuracy (%)')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=norm_hex_colors, labels=norm_legend_labels)
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(7, 2.625),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            strip_text=element_blank(),
            strip_background=element_blank(),
            legend_position='top',
            legend_key_width=9,
            legend_direction='horizontal',
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color)
        )
)

p.save('plots/mmlu-max-params-normalized.png', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-normalized.svg', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-normalized.pdf', dpi=300, width=7, height=2.625)
print("plot saved as 'plots/mmlu-max-params-normalized.png'")

# ============================================================================
# Plot 2: Max params base models (cloze, accuracy)
# ============================================================================
df_cloze = df_cloze[df_cloze['subject'].isin(x_labels_full.keys())]
df_max_params_cloze = df_cloze[df_cloze['params'] == max_params].copy()

# Filter to only valid masks and drop any NaN
df_max_params_cloze = df_max_params_cloze[df_max_params_cloze['mask'].isin(mask_order)].copy()
df_max_params_cloze = df_max_params_cloze.dropna(subset=['mask', 'subject', 'accuracy'])

# Convert accuracy to percentage
df_max_params_cloze['accuracy'] = df_max_params_cloze['accuracy'] * 100

# Add facet column
df_max_params_cloze['facet'] = df_max_params_cloze['subject'].apply(
    lambda x: 'forget' if x in medical_subjects else 'retain'
)

df_max_params_cloze['subject'] = df_max_params_cloze['subject'].replace(x_labels)
df_max_params_cloze['subject'] = pd.Categorical(
    df_max_params_cloze['subject'],
    categories=medical_order + non_medical_order,
    ordered=True
)
df_max_params_cloze['mask'] = df_max_params_cloze['mask'].replace(legend_labels)
df_max_params_cloze['mask'] = pd.Categorical(
    df_max_params_cloze['mask'],
    categories=[legend_labels[m] for m in ['nomask', 'document', 'mask', 'remove']],
    ordered=True
)
df_max_params_cloze['facet'] = pd.Categorical(
    df_max_params_cloze['facet'],
    categories=['forget', 'retain'],
    ordered=True
)

# Create annotation data for facet labels
max_acc_cloze = df_max_params_cloze['accuracy'].max()
facet_labels_cloze = pd.DataFrame({
    'facet': pd.Categorical(['forget', 'retain'], categories=['forget', 'retain'], ordered=True),
    'label': [r'Forget ($\downarrow$)', r'Retain ($\uparrow$)'],
    'subject': [medical_order[0], non_medical_order[0]],
    'y': [max_acc_cloze * 0.93, max_acc_cloze * 0.93]
})

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_max_params_cloze, aes(x='subject', y='accuracy', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + geom_hline(yintercept=25, color=line_color, size=0.5)
        + geom_text(data=facet_labels_cloze, mapping=aes(x='subject', y='y', label='label'),
                   ha='left', va='center', size=9, family='Helvetica Neue',
                   fontweight='bold', color=text_color,
                   nudge_x=-0.45, inherit_aes=False)
        + facet_wrap('~facet', scales='free_x')
        + scale_y_continuous(name='Accuracy (%)')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=hex_colors, labels=list(legend_labels.values()))
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(7, 2.625),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            strip_text=element_blank(),
            strip_background=element_blank(),
            legend_position='top',
            legend_key_width=9,
            legend_direction='horizontal',
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color)
        )
)

p.save('plots/mmlu-max-params-cloze.png', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-cloze.svg', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-cloze.pdf', dpi=300, width=7, height=2.625)
print("plot saved as 'plots/mmlu-max-params-cloze.png'")

# ============================================================================
# Plot 3: Max params instruct models (NLL)
# ============================================================================
df_nll = df_nll[df_nll['subject'].isin(x_labels_full.keys())]
df_max_params_nll = df_nll[df_nll['params'] == max_params].copy()

# Filter to only valid masks and drop any NaN
df_max_params_nll = df_max_params_nll[df_max_params_nll['mask'].isin(mask_order)].copy()
df_max_params_nll = df_max_params_nll.dropna(subset=['mask', 'subject', 'avg_nll'])

# Add facet column
df_max_params_nll['facet'] = df_max_params_nll['subject'].apply(
    lambda x: 'forget' if x in medical_subjects else 'retain'
)

df_max_params_nll['subject'] = df_max_params_nll['subject'].replace(x_labels)
df_max_params_nll['subject'] = pd.Categorical(
    df_max_params_nll['subject'],
    categories=medical_order + non_medical_order,
    ordered=True
)
df_max_params_nll['mask'] = df_max_params_nll['mask'].replace(legend_labels)
df_max_params_nll['mask'] = pd.Categorical(
    df_max_params_nll['mask'],
    categories=[legend_labels[m] for m in ['nomask', 'document', 'mask', 'remove']],
    ordered=True
)
df_max_params_nll['facet'] = pd.Categorical(
    df_max_params_nll['facet'],
    categories=['forget', 'retain'],
    ordered=True
)

# Create annotation data for facet labels (top right for NLL)
max_nll = df_max_params_nll['avg_nll'].max()
facet_labels_nll = pd.DataFrame({
    'facet': pd.Categorical(['forget', 'retain'], categories=['forget', 'retain'], ordered=True),
    'label': [r'Forget ($\uparrow$)', r'Retain ($\downarrow$)'],
    'subject': [medical_order[-1], non_medical_order[-1]],
    'y': [max_nll * 0.93, max_nll * 0.93]
})

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_max_params_nll, aes(x='subject', y='avg_nll', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + geom_text(data=facet_labels_nll, mapping=aes(x='subject', y='y', label='label'),
                   ha='right', va='center', size=9, family='Helvetica Neue',
                   fontweight='bold', color=text_color,
                   nudge_x=0.45, inherit_aes=False)
        + facet_wrap('~facet', scales='free_x')
        + scale_y_continuous(name='Negative Log Likelihood')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=hex_colors, labels=list(legend_labels.values()))
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(7, 2.625),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            strip_text=element_blank(),
            strip_background=element_blank(),
            legend_position='top',
            legend_key_width=9,
            legend_direction='horizontal',
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color)
        )
)

p.save('plots/mmlu-max-params-nll.png', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-nll.svg', dpi=300, width=7, height=2.625)
p.save('plots/mmlu-max-params-nll.pdf', dpi=300, width=7, height=2.625)
print("plot saved as 'plots/mmlu-max-params-nll.png'")

if not args.max_params_only:

    # ============================================================================
    # Plot 4: All params base models (cloze, accuracy)
    # ============================================================================
    breaks_cloze = sorted(df_cloze['params'].unique())
    labels_cloze = [f'{int(p / 1e6)}M' for p in breaks_cloze]
    
    # Add chance levels for faceted plot
    df_cloze['chance_level'] = df_cloze['subject'].map(chance_levels)
    hlines_cloze = df_cloze[['subject', 'chance_level']].drop_duplicates()

    p = (
        ggplot(df_cloze, aes(x='params', y='accuracy', color='mask'))
            + geom_line(size=1)
            + geom_point(size=2, stroke=0, alpha=0.9)
            + geom_point(fill="none", stroke=0.5, size=2, color=grid_color)
            + geom_hline(data=hlines_cloze, mapping=aes(yintercept='chance_level'), color=line_color, size=0.5)
            + facet_wrap('~subject', scales='free_y')
            + scale_x_log10(name='parameters', breaks=breaks_cloze, labels=labels_cloze)
            + scale_y_continuous(name='accuracy')
            + scale_color_manual(values=hex_colors, labels=list(legend_labels.values()))
            # + guides(color=guide_legend(nrow=2))
            + base_theme(base_family='Helvetica Neue')
            + theme(
                figure_size=(14, 10),
                panel_grid_major=element_line(size=0.3, color=grid_color),
                panel_grid_minor=element_blank(),
                legend_title=element_blank(),
                legend_position='top',
                strip_background=element_blank(),
                axis_title_x=element_text(color=text_color),
                axis_title_y=element_text(color=text_color),
                axis_text_x=element_text(color=text_color),
                axis_text_y=element_text(color=text_color),
                legend_text=element_text(color=text_color),
                strip_text=element_text(color=text_color),
                plot_background=element_rect(fill=bg_color),
                panel_background=element_rect(fill=bg_color),
                panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
                legend_background=element_rect(fill=bg_color)
            )
    )

    p.save('plots/mmlu-cloze.png', dpi=300, width=8, height=6)
    print("plot saved as 'plots/mmlu-cloze.png'")

    # ============================================================================
    # Plot 5: All params instruct models (accuracy)
    # ============================================================================
    # Add chance levels for faceted plot
    df['chance_level'] = df['subject'].map(chance_levels)
    hlines_df = df[['subject', 'chance_level']].drop_duplicates()
    
    p = (
        ggplot(df, aes(x='params', y='accuracy', color='mask'))
            + geom_line(size=1)
            + geom_point(size=2, stroke=0, alpha=0.9)
            + geom_point(fill="none", stroke=0.5, size=2, color=grid_color)
            + geom_hline(data=hlines_df, mapping=aes(yintercept='chance_level'), color=line_color, size=0.5)
            + facet_wrap('~subject', scales='free_y')
            + scale_x_log10(name='parameters', breaks=breaks, labels=labels)
            + scale_y_continuous(name='accuracy')
            + scale_color_manual(values=hex_colors, labels=list(legend_labels.values()))
            # + guides(color=guide_legend(nrow=2))
            + base_theme(base_family='Helvetica Neue')
            + theme(
                figure_size=(14, 10),
                panel_grid_major=element_line(size=0.3, color=grid_color),
                panel_grid_minor=element_blank(),
                legend_title=element_blank(),
                legend_position='top',
                strip_background=element_blank(),
                axis_title_x=element_text(color=text_color),
                axis_title_y=element_text(color=text_color),
                axis_text_x=element_text(color=text_color),
                axis_text_y=element_text(color=text_color),
                legend_text=element_text(color=text_color),
                strip_text=element_text(color=text_color),
                plot_background=element_rect(fill=bg_color),
                panel_background=element_rect(fill=bg_color),
                panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
                legend_background=element_rect(fill=bg_color)
            )
    )

    p.save('plots/mmlu.png', dpi=300, width=8, height=6)
    print("plot saved as 'plots/mmlu.png'")

    # ========================================================================
    # Plot 6: All params instruct models (NLL)
    # ========================================================================
    breaks_nll = sorted(df_nll['params'].unique())
    labels_nll = [f'{int(p / 1e6)}M' for p in breaks_nll]

    p = (
        ggplot(df_nll, aes(x='params', y='avg_nll', color='mask'))
            + geom_line(size=1)
            + geom_point(size=2, stroke=0, alpha=0.9)
            + geom_point(fill="none", stroke=0.5, size=2, color=grid_color)
            + facet_wrap('~subject', scales='free_y')
            + scale_x_log10(name='parameters', breaks=breaks_nll, labels=labels_nll)
            + scale_y_continuous(name='average NLL (lower is better)')
            + scale_color_manual(values=hex_colors, labels=list(legend_labels.values()))
            # + guides(color=guide_legend(nrow=2))
            + base_theme(base_family='Helvetica Neue')
            + theme(
                figure_size=(14, 10),
                panel_grid_major=element_line(size=0.3, color=grid_color),
                panel_grid_minor=element_blank(),
                legend_title=element_blank(),
                legend_position='top',
                strip_background=element_blank(),
                axis_title_x=element_text(color=text_color),
                axis_title_y=element_text(color=text_color),
                axis_text_x=element_text(color=text_color),
                axis_text_y=element_text(color=text_color),
                legend_text=element_text(color=text_color),
                strip_text=element_text(color=text_color),
                plot_background=element_rect(fill=bg_color),
                panel_background=element_rect(fill=bg_color),
                panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
                legend_background=element_rect(fill=bg_color)
            )
    )

    p.save('plots/mmlu-nll.png', dpi=300, width=8, height=6)
    print("plot saved as 'plots/mmlu-nll.png'")