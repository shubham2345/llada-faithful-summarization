import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from get_log_likelihood import get_log_likelihood

# Setup device and model
device = "cuda"
model = AutoModel.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)

# Load dataset
dataset = []
with open("results/llada/llada_test_summaries_1.jsonl") as f:
    for line in f:
        dataset.append(json.loads(line))

# Prepare batch tokenization
batch_custom_ids = []
batch_prompt_tokens = []
batch_ref_tokens = []
batch_model_tokens = []

print("Tokenizing inputs...")
for row in tqdm(dataset, desc="Tokenizing inputs"):
    custom_id = row["custom_id"]
    source = "Summarize " + row["source"]
    ref_summ = row["ref_summ"]
    model_summ = row["model_summ"]

    # Tokenize (as tensors)
    prompt_tokens = tokenizer(source, return_tensors="pt")["input_ids"][0]
    ref_tokens = tokenizer(ref_summ, return_tensors="pt")["input_ids"][0]
    model_tokens = tokenizer(model_summ, return_tensors="pt")["input_ids"][0]

    # Store
    batch_custom_ids.append(custom_id)
    batch_prompt_tokens.append(prompt_tokens)
    batch_ref_tokens.append(ref_tokens)
    batch_model_tokens.append(model_tokens)

# Compute NLL in mini-batches
batch_size = 16
results = []

print("Computing NLL...")
for i in tqdm(range(0, len(batch_prompt_tokens), batch_size), desc="Computing NLL"):
    batch_prompts = batch_prompt_tokens[i:i+batch_size]
    batch_refs = batch_ref_tokens[i:i+batch_size]
    batch_models = batch_model_tokens[i:i+batch_size]
    batch_ids = batch_custom_ids[i:i+batch_size]

    # Loop through batch (assuming get_log_likelihood works one example at a time)
    for j, (prompt, ref, model_summ, cid) in enumerate(zip(batch_prompts, batch_refs, batch_models, batch_ids)):
        try:
            # Move inputs to device
            prompt = prompt.to(device)
            ref = ref.to(device)
            model_summ = model_summ.to(device)

            # Compute NLLs
            nll_ref = get_log_likelihood(model, prompt, ref, mc_num=128)
            nll_model = get_log_likelihood(model, prompt, model_summ, mc_num=128)
            nll_diff = nll_model - nll_ref

            # Store result
            results.append({
                "custom_id": cid,
                "NLL_Model": nll_model,
                "NLL_Ref": nll_ref,
                "NLL_Diff": nll_diff
            })

        except Exception as e:
            print(f"Error on example {cid}: {e}")
            continue

# Save as CSV
df = pd.DataFrame(results)
filename = "results/llada/llada_test_summaries_1_nll.csv"
df.to_csv(filename, index=False)

print(f"\nDone! Results saved to {filename}")