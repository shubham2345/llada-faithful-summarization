import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
from models_config import MODEL_CONFIGS

torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPO Fine-tuning Script (converted from faithsum-dpo.ipynb)"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train.jsonl",
        help="Path to the training JSONL file."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama-1b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type to use; selects corresponding tokenizer and SFT checkpoint from MODEL_CONFIGS."
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="prompts",
        help="Directory containing system prompt text files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dpo_checkpoints",
        help="Directory to output DPO checkpoints and results."
    )
    return parser.parse_args()


################################################################################
# Step A: Load system prompts from a local text files directory
################################################################################
def load_prompts(prompts_dir):
    """
    Load system prompts from files in a prompts directory.
    
    Expected files:
      - samsum_sys_prompt.txt
      - cnn_sys_prompt.txt
      - xsum_sys_prompt.txt
    """
    domain_map = {
        "samsum": "samsum_sys_prompt.txt",
        "cnn": "cnn_sys_prompt.txt",
        "xsum": "xsum_sys_prompt.txt",
    }
    prompts_dict = {}
    for domain, fname in domain_map.items():
        path = os.path.join(prompts_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                prompts_dict[domain] = f.read().strip()
        else:
            prompts_dict[domain] = ""  # fallback if file not found
    return prompts_dict

def get_domain_from_custom_id(custom_id):
    """
    Determine the domain from a custom ID using a simple heuristic.
    """
    cid_lower = custom_id.lower()
    if "samsum" in cid_lower:
        return "samsum"
    elif "cnn" in cid_lower:
        return "cnn"
    elif "xsum" in cid_lower:
        return "xsum"
    else:
        return "samsum"  # fallback

################################################################################
# Step B: Build final conversational DPO triplets
################################################################################

def conversation_to_text(conversation):
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])

def create_dpo_triplets(example, prompts_dict):
    """
    Create DPO triplets from an example in the required conversational format.

    Expected input example keys:
      - custom_id: used to determine domain for system prompt.
      - source: the article or input text.
      - ref_summ: the reference (good) summary.
      - model_summ: the model-generated (worse) summary.
      
    Returns a dict with keys:
      - "prompt": conversation prompt (system and user messages).
      - "chosen": full conversation for the chosen (good) response.
      - "rejected": full conversation for the rejected (worse) response.
    """
    domain = get_domain_from_custom_id(example["custom_id"])
    system_prompt = prompts_dict.get(domain, "")
    user_content = f"Article:\n{example['source']}\n\nPlease summarize:"
    
    # Define the conversation prompt (without the assistant reply)
    prompt_conv = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_content}
    ]
    # Full conversations include the assistant reply
    chosen_conv = prompt_conv + [{"role": "assistant", "content": example["ref_summ"]}]
    rejected_conv = prompt_conv + [{"role": "assistant", "content": example["model_summ"]}]
    
    return {
         "prompt": conversation_to_text(prompt_conv),
         "chosen": conversation_to_text(chosen_conv),
         "rejected": conversation_to_text(rejected_conv)
    }

################################################################################
# Step C: Main Data Preparation
################################################################################
def prepare_dataset_for_dpo(train_jsonl_path, prompts_dir):
    """
    Prepare the dataset for DPO training.
    
    Steps:
      1) Load the JSONL data.
      2) Load system prompts.
      3) Create final "chosen" and "rejected" fields in conversational format.
      4) Return a Hugging Face Dataset with these fields.
    """
    dataset = load_dataset("json", data_files=train_jsonl_path, split="train")
    print("Sample before formatting:", dataset[0])
    
    prompts_dict = load_prompts(prompts_dir)
    
    original_cols = dataset.column_names
    dataset = dataset.map(
        lambda x: create_dpo_triplets(x, prompts_dict),
        remove_columns=original_cols
    )
    print("Sample after formatting:", dataset[0])
    return dataset

def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(42)
    
    ################################################################################
    # Data Preparation
    ################################################################################
    print("Loading and preparing dataset from:", args.train_data_path)
    final_dataset = prepare_dataset_for_dpo(args.train_data_path, args.prompts_dir)
    
    # Optionally, limit the dataset size (here: first 5000 examples)
    # final_dataset = final_dataset.select(range(min(5000, len(final_dataset))))
    
    print("Dataset ready for DPO. Fields:", final_dataset.column_names)
    print("Number of samples:", len(final_dataset))
    print("Sample example:", final_dataset[0])
    
    ################################################################################
    # Model and Tokenizer Loading via MODEL_CONFIGS
    ################################################################################

    model_config = MODEL_CONFIGS[args.model_type]
    sft_checkpoint = model_config["finetuned"]
    tokenizer_name = model_config["tokenizer"]

    print(f"Loading SFT checkpoint from: {sft_checkpoint}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify the loaded PEFT configuration
    peft_config = model.peft_config
    print("Loaded PEFT configuration:")
    print(peft_config)

    # Define LoRA configuration (should match your SFT parameters)
    lora_config = LoraConfig(
        r=128,  # As per your SFT parameters
        lora_alpha=256,
        target_modules=['q_proj', 'down_proj', 'gate_proj', 'o_proj', 'k_proj', 'up_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    ################################################################################
    # DPO Training Setup
    ################################################################################
    dpo_config = DPOConfig(
        beta=0.1,                        # Trade-off between reward and KL
        loss_type="sigmoid",             # Use sigmoid loss for DPO
        max_length=512,                  # Maximum sequence length (input + generation)
        max_prompt_length=256,           # Maximum prompt length
        reference_free=True,             # No separate reference model used
        output_dir=args.output_dir,      # Directory for saving checkpoints/results
        per_device_train_batch_size=4,   # Adjust based on available GPU memory
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_steps=200,
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False
    )

    # Initialize the DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        peft_config=lora_config,
        args=dpo_config,
        train_dataset=final_dataset,
        tokenizer=tokenizer
    )

    ################################################################################
    # Start Training
    ################################################################################
    print("Starting DPO training...")
    trainer.train()
    print("DPO training complete.")

if __name__ == "__main__":
    main()