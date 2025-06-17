import argparse
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from models_config import MODEL_CONFIGS

def load_prompts(prompts_dir):
    """
    Loads system prompt text files from 'prompts_dir' and returns a dict.
    Update domain_map to reflect your actual file names.
    """
    domain_map = {
        "samsum": "samsum_sys_prompt.txt",
        "cnn": "cnn_sys_prompt.txt",
        "xsum": "xsum_sys_prompt.txt",
    }
    prompts = {}
    for domain, fname in domain_map.items():
        full_path = os.path.join(prompts_dir, fname)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                prompts[domain] = f.read().strip()
        else:
            prompts[domain] = ""
    return prompts

def get_domain_from_custom_id(custom_id):
    """
    Returns a domain key like "samsum", "cnn", or "xsum" 
    based on custom_id. Adjust logic to match your naming conventions.
    """
    custom_id_lower = custom_id.lower()
    if "samsum" in custom_id_lower:
        return "samsum"
    elif "cnn" in custom_id_lower:
        return "cnn"
    elif "xsum" in custom_id_lower:
        return "xsum"
    else:
        return "samsum"  # fallback or raise an error

def create_dpo_pairs(example, prompts_dict, use_model_summ=False):
    """
    Combine system prompts + source text + ref_summ (chosen) + model_summ (rejected).
    
    Example input keys:
        - custom_id
        - source
        - ref_summ
        - model_summ (optional if use_model_summ is True)
    """
    domain = get_domain_from_custom_id(example["custom_id"])
    system_prompt = prompts_dict.get(domain, "")

    chosen_text = (
        system_prompt
        + "\n\n" + example["source"]
        + "\n\n" + example["ref_summ"]
    )

    if use_model_summ and "model_summ" in example and example["model_summ"]:
        rejected_text = (
            system_prompt
            + "\n\n" + example["source"]
            + "\n\n" + example["model_summ"]
        )
    else:
        # fallback if no model_summ
        rejected_text = (
            system_prompt
            + "\n\n" + example["source"]
            + "\n\n" + "NO_REJECTED_SUMMARY"
        )

    return {
        "chosen_text": chosen_text,
        "rejected_text": rejected_text
    }

def tokenize_for_dpo(example, tokenizer, max_length=512):
    chosen_enc = tokenizer(
        example["chosen_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    rejected_enc = tokenizer(
        example["rejected_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    example["input_ids_chosen"] = chosen_enc["input_ids"]
    example["attention_mask_chosen"] = chosen_enc["attention_mask"]
    example["input_ids_rejected"] = rejected_enc["input_ids"]
    example["attention_mask_rejected"] = rejected_enc["attention_mask"]
    return example

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for DPO fine-tuning, with domain-based system prompts.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to train.jsonl")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test.jsonl")
    parser.add_argument("--prompts_dir", type=str, required=True,
                        help="Directory containing domain-specific system prompt text files.")
    parser.add_argument("--model_key", type=str, required=True,
                        help="Key to select which model to load from MODEL_CONFIGS.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the tokenized dataset.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for each example.")
    parser.add_argument("--use_model_summ", action="store_true",
                        help="If set, script expects 'model_summ' for the rejected text.")
    args = parser.parse_args()

    # Check the model_key in MODEL_CONFIGS
    if args.model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model key {args.model_key} not found in MODEL_CONFIGS")
    
    # Retrieve the tokenizer name from MODEL_CONFIGS
    model_name = MODEL_CONFIGS[args.model_key]["tokenizer"]

    # 1) Load system prompts
    prompts_dict = load_prompts(args.prompts_dir)

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load train/test data from JSONL
    raw_train = load_dataset("json", data_files={"train": args.train_path})["train"]
    raw_test = load_dataset("json", data_files={"test": args.test_path})["test"]

    # 4) Create chosen/rejected pairs
    train_dataset = raw_train.map(
        lambda x: create_dpo_pairs(x, prompts_dict, use_model_summ=args.use_model_summ),
        batched=False
    )
    test_dataset = raw_test.map(
        lambda x: create_dpo_pairs(x, prompts_dict, use_model_summ=args.use_model_summ),
        batched=False
    )

    # 5) Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_for_dpo(x, tokenizer, args.max_length),
        batched=False
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_for_dpo(x, tokenizer, args.max_length),
        batched=False
    )

    # 6) Convert to torch format
    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids_chosen", "attention_mask_chosen",
            "input_ids_rejected", "attention_mask_rejected"
        ]
    )
    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids_chosen", "attention_mask_chosen",
            "input_ids_rejected", "attention_mask_rejected"
        ]
    )

    # 7) Save
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)
    print(f"DPO-ready dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()