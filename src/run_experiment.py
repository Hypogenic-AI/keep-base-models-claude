"""Main experiment runner: generates text, evaluates it, and analyzes results."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

def main():
    start_time = time.time()

    print("=" * 70)
    print("KEEPING BASE MODEL DISTRIBUTIONS - FULL EXPERIMENT PIPELINE")
    print("=" * 70)

    # Environment info
    import torch
    import numpy as np
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # Save config
    from config import (BASE_MODEL, ALIGNED_MODEL, STYLES, TOPICS, ALPHA_VALUES,
                        MAX_NEW_TOKENS, TEMPERATURE, TOP_P, NUM_SAMPLES_PER_TASK, SEED)

    config = {
        "base_model": BASE_MODEL,
        "aligned_model": ALIGNED_MODEL,
        "styles": list(STYLES.keys()),
        "num_topics": len(TOPICS),
        "alpha_values": ALPHA_VALUES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_samples_per_task": NUM_SAMPLES_PER_TASK,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Generation
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATION")
    print("=" * 70)
    from generate import run_generation_experiment
    run_generation_experiment()

    gen_time = time.time() - start_time
    print(f"\nGeneration completed in {gen_time/60:.1f} minutes")

    # Phase 2: Evaluation
    print("\n" + "=" * 70)
    print("PHASE 2: EVALUATION")
    print("=" * 70)
    from evaluate import run_evaluation
    run_evaluation()

    eval_time = time.time() - start_time - gen_time
    print(f"\nEvaluation completed in {eval_time/60:.1f} minutes")

    # Phase 3: Analysis
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)
    from analyze import run_analysis
    run_analysis()

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE. Total time: {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
