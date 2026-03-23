"""Configuration for the style preservation experiments."""

import os

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-3B"
ALIGNED_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Generation parameters
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95
NUM_SAMPLES_PER_TASK = 3
NUM_INTERP_SAMPLES = 2  # Fewer samples for distribution arithmetic (slower)
SEED = 42

# Distribution arithmetic interpolation coefficients
# α=0 means pure base, α=1 means pure aligned
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Style definitions: name -> description for prompting
STYLES = {
    "hemingway": "Ernest Hemingway (short declarative sentences, sparse prose, understated emotion)",
    "shakespeare": "William Shakespeare (iambic pentameter, archaic English, metaphor-rich)",
    "poe": "Edgar Allan Poe (gothic atmosphere, dark imagery, ornate vocabulary)",
    "austen": "Jane Austen (ironic wit, formal Regency-era English, social commentary)",
    "tolkien": "J.R.R. Tolkien (epic grandeur, detailed world-building, archaic diction)",
    "academic": "An academic research paper (formal, hedged claims, citations, passive voice)",
    "noir": "Hard-boiled film noir narration (cynical first-person, vivid metaphors, urban setting)",
}

# Topics to write about in each style
TOPICS = [
    "A person arriving at an old house for the first time",
    "The experience of watching a sunset over the ocean",
    "A conversation between two strangers on a train",
]

# OpenAI evaluation config
OPENAI_MODEL = "gpt-4.1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Paths
RESULTS_DIR = "results"
GENERATIONS_DIR = f"{RESULTS_DIR}/generations"
EVALUATIONS_DIR = f"{RESULTS_DIR}/evaluations"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
