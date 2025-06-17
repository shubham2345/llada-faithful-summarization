# LLaDA-Faithful-Summarization

**Faithfulness Evaluation of Diffusion-Based vs. Autoregressive Language Models for Abstractive Summarization**

This repository contains the codebase, evaluation scripts, and data pipelines for comparing the faithfulness of LLaDA, a diffusion-based language model, with several autoregressive models (ARMs) such as LLaMA and SmolLM, using both zero-shot and fine-tuned settings.

---

## 🔍 Overview

Modern language models excel at fluency but often struggle with **factual faithfulness** in abstractive summarization. This project explores whether **LLaDA** can serve as a more faithful alternative to ARMs by leveraging a **diffusion-style denoising decoder** and **bidirectional context modeling**.

We build a **preference-labeled dataset** using span-level hallucination detection and fine-tune ARMs with **Direct Preference Optimization (DPO)**. Evaluations are conducted on CNN/DailyMail, SAMSum, and XSum using AlignScore and BERTScore.

---

## 📁 Repository Structure

```plaintext
├── evals/
│   ├── negative_log_likelihood.py     # NLL computation
│   ├── score.py                       # AlignScore & BERTScore evaluator
│   ├── generation/
│   │   ├── arm.py                     # Generation pipeline for ARMs
│   │   └── llada.py                   # Generation pipeline for LLaDA
│   ├── prompts/
│   │   ├── cnn_sys_prompt.txt
│   │   ├── samsum_sys_prompt.txt
│   │   └── xsum_sys_prompt.txt
│   ├── results/
│   │   ├── dpo/
│   │   └── llada/
│   └── aggregated_results_per_dataset.csv  # Final evaluation metrics
│
├── sft/
│   ├── preprocess.py                 # Prepares SFT training data
│   └── train.py                      # SFT + DPO training script
│
├── models_config.py                  # Model initialization/config
└── README.md