"""Evaluate generated text for style fidelity, diversity, and quality using GPT-4.1 and automated metrics."""

import json
import os
import sys
import time
import numpy as np
from collections import Counter
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    STYLES, TOPICS, OPENAI_MODEL, OPENAI_API_KEY,
    GENERATIONS_DIR, EVALUATIONS_DIR
)


def compute_distinct_ngrams(texts, n=2):
    """Compute distinct n-gram ratio (type/token ratio for n-grams)."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_type_token_ratio(texts):
    """Compute type-token ratio across all texts."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def compute_self_bleu_approx(texts, n=4):
    """Approximate self-BLEU: average n-gram overlap between pairs of texts.
    Lower = more diverse."""
    if len(texts) < 2:
        return 0.0

    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            tokens_i = texts[i].lower().split()
            tokens_j = texts[j].lower().split()
            ngrams_i = set(tuple(tokens_i[k:k+n]) for k in range(len(tokens_i) - n + 1))
            ngrams_j = set(tuple(tokens_j[k:k+n]) for k in range(len(tokens_j) - n + 1))
            if ngrams_i and ngrams_j:
                overlap = len(ngrams_i & ngrams_j) / max(len(ngrams_i), len(ngrams_j))
                overlaps.append(overlap)
    return np.mean(overlaps) if overlaps else 0.0


def compute_avg_length(texts):
    """Average length in words."""
    lengths = [len(t.split()) for t in texts]
    return np.mean(lengths) if lengths else 0.0


def evaluate_style_with_gpt(client, text, style_name, style_desc, topic):
    """Use GPT-4.1 to evaluate style fidelity and quality."""
    prompt = f"""You are an expert literary critic. Evaluate the following text on two dimensions.

TARGET STYLE: {style_desc}
TOPIC: {topic}

TEXT TO EVALUATE:
\"\"\"
{text[:800]}
\"\"\"

Rate on a scale of 1-10:
1. STYLE_FIDELITY: How well does this text capture the distinctive voice, vocabulary, sentence structure, and literary techniques of {style_name}? (1=not at all, 10=indistinguishable from the real thing)
2. TEXT_QUALITY: How coherent, well-written, and engaging is this text regardless of style? (1=incoherent, 10=excellent)

Respond ONLY with JSON: {{"style_fidelity": <int>, "text_quality": <int>, "style_reasoning": "<brief explanation>"}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON from response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"  GPT evaluation error: {e}")
        return {"style_fidelity": -1, "text_quality": -1, "style_reasoning": f"error: {str(e)}"}


def run_evaluation():
    """Run full evaluation on generated texts."""
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)

    print("=" * 60)
    print("STYLE PRESERVATION EXPERIMENT - EVALUATION PHASE")
    print("=" * 60)

    # Load generations
    gen_path = f"{GENERATIONS_DIR}/all_generations.json"
    with open(gen_path) as f:
        all_generations = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    all_metrics = {}
    gpt_call_count = 0

    for style_name in all_generations:
        style_desc = STYLES[style_name]
        all_metrics[style_name] = {}

        for topic in all_generations[style_name]:
            task_data = all_generations[style_name][topic]
            all_metrics[style_name][topic] = {}

            for condition, texts in task_data.items():
                if not texts or all(not t.strip() for t in texts):
                    continue

                texts = [t for t in texts if t.strip()]

                # Automated diversity metrics
                metrics = {
                    "distinct_1": compute_distinct_ngrams(texts, 1),
                    "distinct_2": compute_distinct_ngrams(texts, 2),
                    "distinct_3": compute_distinct_ngrams(texts, 3),
                    "type_token_ratio": compute_type_token_ratio(texts),
                    "self_bleu_4": compute_self_bleu_approx(texts, 4),
                    "avg_length": compute_avg_length(texts),
                    "num_samples": len(texts),
                }

                # GPT-4.1 style evaluation (sample 2 per condition to manage costs)
                if client:
                    gpt_scores = []
                    for text in texts[:2]:
                        if not text.strip():
                            continue
                        score = evaluate_style_with_gpt(client, text, style_name, style_desc, topic)
                        gpt_scores.append(score)
                        gpt_call_count += 1
                        if gpt_call_count % 20 == 0:
                            print(f"  GPT calls: {gpt_call_count}")
                        time.sleep(0.2)  # Rate limiting

                    valid_scores = [s for s in gpt_scores if s.get("style_fidelity", -1) > 0]
                    if valid_scores:
                        metrics["style_fidelity_mean"] = np.mean([s["style_fidelity"] for s in valid_scores])
                        metrics["text_quality_mean"] = np.mean([s["text_quality"] for s in valid_scores])
                        metrics["style_fidelity_scores"] = [s["style_fidelity"] for s in valid_scores]
                        metrics["text_quality_scores"] = [s["text_quality"] for s in valid_scores]
                        metrics["style_reasoning"] = [s.get("style_reasoning", "") for s in valid_scores]

                all_metrics[style_name][topic][condition] = metrics
                print(f"  {style_name}/{condition}: D2={metrics['distinct_2']:.3f}, "
                      f"SB4={metrics['self_bleu_4']:.3f}, "
                      f"SF={metrics.get('style_fidelity_mean', 'N/A')}")

    # Save all metrics
    with open(f"{EVALUATIONS_DIR}/all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nEvaluation complete. {gpt_call_count} GPT calls made.")
    print(f"Results saved to {EVALUATIONS_DIR}/all_metrics.json")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
