import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from models_config import MODEL_CONFIGS
from peft import PeftModel
import re
import unicodedata

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_key", type=str, default=None,
                        help="Key to select from MODEL_CONFIGS if you want to load from the dictionary.")
    parser.add_argument("--version", type=str, default="finetuned",
                        help="Which version of the model to load: 'finetuned' (SFT) or 'contrastive'.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Path or HF hub ID for the tokenizer (base/original model).")
    parser.add_argument("--finetuned_model_path", type=str, default=None,
                        help="Path or HF hub ID for the fine-tuned model weights.")
    
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test file (line-delimited JSON).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save annotated samples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling.")
    parser.add_argument("--method", type=str, default="dpo",
                        help="Method used for fine-tuning the models")
    
    return parser.parse_args()

def load_prompt(prompt_path):
    """Utility function to load a text file into a string."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_text(text, lowercase=False):
    """Normalize text by converting to ASCII, removing extra whitespace, and replacing newlines."""
    # Normalize unicode to ASCII (removes accents, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Replace carriage returns and newlines with a space
    text = text.replace('\r', ' ').replace('\n', ' ')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    if lowercase:
        text = text.lower()
    return text

def main():
    args = parse_args()
    
    if args.model_key is not None:
        if args.model_key not in MODEL_CONFIGS:
            raise ValueError(f"model_key '{args.model_key}' not found in MODEL_CONFIGS.")
        config = MODEL_CONFIGS[args.model_key]
        tokenizer_name_or_path = config["tokenizer"]
        
        if args.version == "contrastive":
            finetuned_model_path = config.get("contrastive")
            if finetuned_model_path is None:
                raise ValueError(F"contrastive checkpoint mismatch with")
        elif args.version == "dpo":
            finetuned_model_path = config.get("dpo")
            if finetuned_model_path is None:
                raise ValueError("dpo checkpoint mismatch")
        else:
            finetuned_model_path = config["finetuned"]
    else:
        # Otherwise, use explicitly provided paths
        tokenizer_name_or_path = args.tokenizer_name_or_path
        finetuned_model_path = args.finetuned_model_path
    
    if not tokenizer_name_or_path:
        raise ValueError("No tokenizer path provided. Use --model_key or --tokenizer_name_or_path.")
    if not finetuned_model_path:
        raise ValueError("No finetuned model path provided. Use --model_key or --finetuned_model_path.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        tokenizer_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True          
    )
    
    model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    model.eval()
    torch.cuda.empty_cache()

    # ---------------------------
    # LOAD ALL PROMPTS ONCE
    # ---------------------------

    cnn_prompt = load_prompt("prompts/cnn_sys_prompt.txt")
    samsum_prompt = load_prompt("prompts/samsum_sys_prompt.txt")
    xsum_prompt = load_prompt("prompts/xsum_sys_prompt.txt")

    test_data = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: 
                test_data.append(json.loads(line))
    
    # test_data = test_data[25:30]
    annotated_data = []
    
    for sample in tqdm(test_data, desc="Generating Summaries"):
        custom_id = sample["custom_id"]
        conversation_text = normalize_text(sample["source"])
        
        if "cnn" in custom_id.lower():
            system_prompt = cnn_prompt
        elif "samsum" in custom_id.lower():
            system_prompt = samsum_prompt
        elif "xsum" in custom_id.lower():
            system_prompt = xsum_prompt
        else:
            system_prompt = "Summarize the following conversation:\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text},
        ]
        
        # print(messages)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenizer=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True, 
                #num_beams=5, #increases the randomness
                # repetition_penalty=3.0,
                # no_repeat_ngram_size=5,
                eos_token_id=tokenizer.eos_token_id
                # early_stopping=True,
                # length_penalty=0.8
            )
        
        generated_text = tokenizer.decode(output_ids[0, inputs.size(1):], skip_special_tokens=True)
        
        if "Summary:" in generated_text:
            generated_text = generated_text.split("Summary:", 1)[-1].strip()
        
        # sample["model_summary"] = generated_text
        # annotated_data.append(sample)

        output_sample = {
            "custom_id": sample["custom_id"],
            "source": sample["source"],
            "ref_summ": sample.get("ref_summ", ""),
            "model": args.model_key,
            "method": args.method,
            "model_summ": generated_text
        }
        annotated_data.append(output_sample)

    output_file = os.path.join(args.output_dir, "annotated_test.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in annotated_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Done! Annotated file saved at: {output_file}")

if __name__ == "__main__":
    main()