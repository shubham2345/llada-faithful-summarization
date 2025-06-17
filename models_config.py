MODEL_CONFIGS = {
    "llama-8b": {
        "tokenizer": "meta-llama/Meta-Llama-3-8B-Instruct",
        "finetuned": "checkpoints/CEP-0.0-Llama-3.1-8B-Instruct",
        "contrastive": "contrastive_checkpoints/llama-8b/checkpoint-2000",
        "dpo": "dpo_checkpoints/llama8b/checkpoint-27975"
    },
    "olmo2_7b": {
        "tokenizer": "allenai/OLMo-2-1124-7B-Instruct",
        "finetuned": "checkpoints/CEP-0.0-OLMo-2-1124-7B-Instruct"
    },
    "smolLM2": {
        "tokenizer": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "finetuned": "checkpoints/CEP-0.0-SmolLM2-1.7B-Instruct",
        "dpo": "dpo_checkpoints/smolLM2/checkpoint-27975"
    },
    "llama-1b": {
        "tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
        "finetuned": "checkpoints/CEP-0.0-Llama-3.2-1B-Instruct", 
        "contrastive": "contrastive_checkpoints/llama-1b/checkpoint-1000",
        "dpo": "dpo_checkpoints/llama1b/checkpoint-23500"
    }
}