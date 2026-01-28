"""
Annotate documents with medical/non-medical labels using Claude batch API.
Unlike filter-test.py which keeps only one class, this script keeps ALL documents
and adds a 'medical' field (True/False) for document-level classification.

Usage:
    python annotate-docs-batch.py --split test_target
    python annotate-docs-batch.py --split test_target --resume batch_id1,batch_id2
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, required=True, help="name of the jsonl file (without .jsonl extension)")
parser.add_argument("--data_path", type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument("--prompt", type=str, default=os.path.join(os.path.dirname(__file__), "..", "document_prompt.txt"))
parser.add_argument("--max_tokens", type=int, default=10)
parser.add_argument("--chunk_size", type=int, default=7500, help="Max prompts per batch request (max 100k)")
parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between polling for batch status")
parser.add_argument("--resume", type=str, default=None, help="Comma-separated batch IDs to resume from")
parser.add_argument("--model", type=str, default="claude-sonnet-4-5", help="Model to use for classification")
args = parser.parse_args()

# load prompt
with open(args.prompt, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read().strip()
print(f"Loaded prompt from {args.prompt}")

# input/output paths
input_path = os.path.join(args.data_path, f"{args.split}.jsonl")
output_path = os.path.join(args.data_path, f"{args.split}_annotated.jsonl")
batch_ids_path = os.path.join(args.data_path, f"{args.split}_batch_ids.txt")

if not os.path.exists(input_path):
    print(f"Error: input file {input_path} does not exist")
    sys.exit(1)

# load input documents
print(f"Loading documents from {input_path}")
documents = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        documents.append(json.loads(line))

print(f"Loaded {len(documents)} documents to process")

# initialize client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_BATCH_API_KEY"))


def create_batch_requests(docs: list[dict], start_idx: int) -> list[Request]:
    """Create batch request objects for a list of documents."""
    requests = []
    for i, doc in enumerate(docs):
        text = doc.get('text', '')
        # Truncate very long documents to avoid token limits
        if len(text) > 50000:
            text = text[:50000] + "..."
        requests.append(
            Request(
                custom_id=f"doc-{start_idx + i}",
                params=MessageCreateParamsNonStreaming(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"Document: {text}"
                    }]
                )
            )
        )
    return requests


def submit_batch(requests: list[Request]) -> str:
    """Submit a batch and return the batch ID."""
    message_batch = client.messages.batches.create(requests=requests)
    return message_batch.id


def wait_for_batch(batch_id: str) -> None:
    """Poll until batch processing completes."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        print(f"  Batch {batch_id}: {status} | "
              f"succeeded={counts.succeeded}, errored={counts.errored}, "
              f"processing={counts.processing}, expired={counts.expired}, canceled={counts.canceled}")

        if status == "ended":
            break

        time.sleep(args.poll_interval)


def get_batch_results(batch_id: str) -> dict[str, tuple[str | None, str]]:
    """Retrieve results from a completed batch. Returns dict mapping custom_id to (response_text, status).
    Status is one of: "success", "refusal", "error"
    """
    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            message = result.result.message
            stop_reason = message.stop_reason

            if stop_reason == "refusal" or not message.content:
                results[custom_id] = (None, "refusal")
            else:
                text = ""
                for block in message.content:
                    if block.type == "text":
                        text += block.text
                results[custom_id] = (text.strip(), "success")
        else:
            results[custom_id] = (None, "error")
    return results


# Handle resume mode
if args.resume:
    print(f"Resuming from provided batch IDs: {args.resume}")
    batch_ids = [bid.strip() for bid in args.resume.split(',')]
else:
    # Submit new batches
    num_chunks = (len(documents) + args.chunk_size - 1) // args.chunk_size
    print(f"Splitting {len(documents)} documents into {num_chunks} batches of up to {args.chunk_size}")

    batch_ids = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(documents))
        chunk_docs = documents[start:end]

        print(f"Submitting batch {chunk_idx + 1}/{num_chunks} (docs {start}-{end-1})...")
        requests = create_batch_requests(chunk_docs, start)
        batch_id = submit_batch(requests)
        batch_ids.append(batch_id)
        print(f"  Created batch: {batch_id}")

    # Save batch IDs for potential resume
    with open(batch_ids_path, 'w') as f:
        f.write(','.join(batch_ids))
    print(f"Saved batch IDs to {batch_ids_path}")

print(f"\nWaiting for {len(batch_ids)} batches to complete...")

# wait for all batches to complete
for i, batch_id in enumerate(batch_ids):
    print(f"\nWaiting for batch {i + 1}/{len(batch_ids)} ({batch_id})...")
    wait_for_batch(batch_id)

print("\nAll batches completed. Retrieving results...")

# collect all results
all_results = {}
for i, batch_id in enumerate(batch_ids):
    print(f"Retrieving results from batch {i + 1}/{len(batch_ids)} ({batch_id})...")
    batch_results = get_batch_results(batch_id)
    all_results.update(batch_results)
    print(f"  Got {len(batch_results)} results")

print(f"\nTotal results retrieved: {len(all_results)}")

# process responses and annotate documents
annotated_docs = []
num_medical = 0
num_non_medical = 0
num_refusal = 0
num_error = 0

for i, doc in enumerate(documents):
    custom_id = f"doc-{i}"
    result = all_results.get(custom_id)

    # Create a copy of the doc with annotation
    annotated_doc = doc.copy()

    if result is None:
        num_error += 1
        annotated_doc['medical'] = None
        annotated_doc['annotation_status'] = 'missing'
    else:
        response_text, status = result

        if status == "refusal":
            num_refusal += 1
            annotated_doc['medical'] = None
            annotated_doc['annotation_status'] = 'refusal'
        elif status != "success" or response_text is None:
            num_error += 1
            annotated_doc['medical'] = None
            annotated_doc['annotation_status'] = 'error'
        else:
            is_medical = response_text.upper().startswith("YES")
            annotated_doc['medical'] = is_medical
            annotated_doc['annotation_status'] = 'success'

            if is_medical:
                num_medical += 1
            else:
                num_non_medical += 1

    annotated_docs.append(annotated_doc)

print(f"\nAnnotation complete:")
print(f"  Medical: {num_medical}")
print(f"  Non-medical: {num_non_medical}")
print(f"  Refusal: {num_refusal}")
print(f"  Errors: {num_error}")

# write output
with open(output_path, 'w', encoding='utf-8') as f:
    for doc in annotated_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

print(f"\n=== Done ===")
print(f"Annotated {len(annotated_docs)} documents")
print(f"Output written to: {output_path}")
