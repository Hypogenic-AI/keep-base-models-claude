"""Generate text in various styles using base model, aligned model, and distribution arithmetic."""

import json
import os
import sys
import time
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_MODEL, ALIGNED_MODEL, STYLES, TOPICS,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, NUM_SAMPLES_PER_TASK,
    NUM_INTERP_SAMPLES, ALPHA_VALUES, SEED, GENERATIONS_DIR
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models():
    """Load base and aligned models on separate GPUs."""
    print(f"Loading base model: {BASE_MODEL}")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    base_model.eval()

    print(f"Loading aligned model: {ALIGNED_MODEL}")
    aligned_tokenizer = AutoTokenizer.from_pretrained(ALIGNED_MODEL, trust_remote_code=True)
    aligned_model = AutoModelForCausalLM.from_pretrained(
        ALIGNED_MODEL, torch_dtype=torch.float16, device_map="cuda:1", trust_remote_code=True
    )
    aligned_model.eval()

    # Use the base tokenizer for both (same vocabulary)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    return base_model, aligned_model, base_tokenizer


def make_base_prompt(style_name, style_desc, topic):
    """Create a prompt for the base model (completion-style)."""
    return (
        f"The following is a passage written in the style of {style_desc}.\n"
        f"Topic: {topic}\n\n"
        f"---\n\n"
    )


def make_instruct_prompt(style_name, style_desc, topic):
    """Create a prompt for the instruct model (chat-style)."""
    return (
        f"Write a short passage (about 150 words) in the style of {style_desc}. "
        f"The topic is: {topic}. "
        f"Capture the distinctive voice, vocabulary, and sentence structure of this style. "
        f"Write only the passage, no commentary."
    )


@torch.no_grad()
def generate_standard(model, tokenizer, prompt, device, num_samples=1, is_chat=False):
    """Generate text from a single model."""
    results = []
    for _ in range(num_samples):
        if is_chat:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(generated.strip())
    return results


@torch.no_grad()
def generate_distribution_arithmetic(base_model, aligned_model, tokenizer, prompt_base, prompt_aligned, alpha, num_samples=1):
    """
    Generate using distribution arithmetic with KV caching for speed.
    logits = (1-α) * base_logits + α * aligned_logits
    α=0 => pure base model, α=1 => pure aligned model
    """
    results = []
    base_device = next(base_model.parameters()).device
    aligned_device = next(aligned_model.parameters()).device

    for _ in range(num_samples):
        base_inputs = tokenizer(prompt_base, return_tensors="pt").to(base_device)
        messages = [{"role": "user", "content": prompt_aligned}]
        aligned_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        aligned_inputs = tokenizer(aligned_text, return_tensors="pt").to(aligned_device)

        # Prefill: get initial KV caches
        base_out = base_model(**base_inputs, use_cache=True)
        aligned_out = aligned_model(**aligned_inputs, use_cache=True)
        base_cache = base_out.past_key_values
        aligned_cache = aligned_out.past_key_values

        base_logits = base_out.logits[:, -1, :].float().to("cpu")
        aligned_logits = aligned_out.logits[:, -1, :].float().to("cpu")

        generated_tokens = []

        for step in range(MAX_NEW_TOKENS):
            # Interpolate in logit space
            mixed_logits = (1 - alpha) * base_logits + alpha * aligned_logits
            mixed_logits = mixed_logits / TEMPERATURE

            # Top-p sampling
            probs = torch.softmax(mixed_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > TOP_P
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, 1)
            next_token_id = sorted_indices[0, idx[0]].item()

            if next_token_id == tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token_id)

            # Feed next token with KV cache (single token forward pass)
            next_base = torch.tensor([[next_token_id]], device=base_device)
            next_aligned = torch.tensor([[next_token_id]], device=aligned_device)

            base_out = base_model(input_ids=next_base, past_key_values=base_cache, use_cache=True)
            aligned_out = aligned_model(input_ids=next_aligned, past_key_values=aligned_cache, use_cache=True)

            base_cache = base_out.past_key_values
            aligned_cache = aligned_out.past_key_values
            base_logits = base_out.logits[:, -1, :].float().to("cpu")
            aligned_logits = aligned_out.logits[:, -1, :].float().to("cpu")

        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append(text.strip())

    return results


def run_generation_experiment():
    """Run the full generation experiment."""
    set_seed(SEED)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)

    print("=" * 60)
    print("STYLE PRESERVATION EXPERIMENT - GENERATION PHASE")
    print("=" * 60)

    base_model, aligned_model, tokenizer = load_models()

    all_results = {}
    total_tasks = len(STYLES) * len(TOPICS)
    task_num = 0

    for style_name, style_desc in STYLES.items():
        all_results[style_name] = {}

        for topic in TOPICS:
            task_num += 1
            print(f"\n[{task_num}/{total_tasks}] Style: {style_name}, Topic: {topic[:50]}...")

            base_prompt = make_base_prompt(style_name, style_desc, topic)
            instruct_prompt = make_instruct_prompt(style_name, style_desc, topic)

            task_results = {}

            # 1. Base model generation
            print("  Generating: base model...")
            task_results["base"] = generate_standard(
                base_model, tokenizer, base_prompt,
                device=next(base_model.parameters()).device,
                num_samples=NUM_SAMPLES_PER_TASK, is_chat=False
            )

            # 2. Aligned model generation
            print("  Generating: aligned model...")
            task_results["aligned"] = generate_standard(
                aligned_model, tokenizer, instruct_prompt,
                device=next(aligned_model.parameters()).device,
                num_samples=NUM_SAMPLES_PER_TASK, is_chat=True
            )

            # 3. Distribution arithmetic at various alpha values
            for alpha in ALPHA_VALUES:
                if alpha == 0.0 or alpha == 1.0:
                    continue  # Already covered by base and aligned
                print(f"  Generating: interpolated α={alpha}...")
                task_results[f"interp_{alpha}"] = generate_distribution_arithmetic(
                    base_model, aligned_model, tokenizer,
                    base_prompt, instruct_prompt,
                    alpha=alpha, num_samples=NUM_INTERP_SAMPLES
                )

            all_results[style_name][topic] = task_results

            # Save incrementally
            with open(f"{GENERATIONS_DIR}/all_generations.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nGeneration complete. Results saved to {GENERATIONS_DIR}/all_generations.json")
    return all_results


if __name__ == "__main__":
    run_generation_experiment()
