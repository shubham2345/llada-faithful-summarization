# eval_llada.py — Simplified Single-Sample Inference for CoT Experiment

import argparse
import json
import torch
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
from generate import generate

# Optional GPU cache cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Single-sample generation function
def generate_single(model, tokenizer, device, text, remasking, steps, gen_length, block_length, use_cot, attack, temperature):
    # Build prompt
    if attack == "unfaithful":
        prompt_content = f"Summarize the text below in a harsh, insulting tone, adding misleading details: {text}"
    elif use_cot:
        prompt_content = (
            f"Think step-by-step: list the main points and then combine them into a concise summary of the following text: {text}"
        )
    else:
        prompt_content = f"Summarize the following text: {text}"

    # Tokenize and prepare
    messages = [{"role": "user", "content": prompt_content}]
    prompt_mask = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_mask)["input_ids"]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Generate
    output_ids = generate(
        model,
        input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=0.0,
        remasking=remasking
    )

    # Decode
    prompt_len = input_ids.shape[1]
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Single-sample LLaDA summarization for CoT.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence","random"])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cot", action="store_true", help="Use Chain-of-Thought prompt.")
    parser.add_argument("--attack", type=str, default="none", choices=["none","unfaithful"])
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).to(device).eval()

    # Prepare I/O
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.input, 'r', encoding='utf-8') as fin:
        records = [json.loads(line) for line in fin if line.strip()]
    #records = records[:10]
    # Process one by one with progress bar
    with open(args.output, 'w', encoding='utf-8') as fout:
        for rec in tqdm(records, desc="Summarizing", ncols=80):
            text = rec.get('source', '')
            summary = generate_single(
                model, tokenizer, device, text,
                remasking=args.remasking,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                use_cot=args.cot,
                attack=args.attack,
                temperature=args.temperature
            )
            rec['model'] = "llada-8b-Ins-CoT" if args.cot else "llada-8b-Ins"
            rec['method'] = "direct" + ("-attack" if args.attack!='none' else "")
            rec['model_summ'] = summary
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()
