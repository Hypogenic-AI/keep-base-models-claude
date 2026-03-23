# Downloaded Datasets

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: UltraFeedback Binarized

### Overview
- **Source**: [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- **Size**: 61,135 train pairs + 2,000 test pairs (~120 MB)
- **Format**: HuggingFace Dataset (chosen/rejected pairs with scores)
- **Task**: DPO preference alignment
- **Splits**: train_prefs (61,135), train_sft (61,135), test_prefs (2,000), test_sft (1,000)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
ds.save_to_disk("datasets/ultrafeedback_binarized")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/ultrafeedback_binarized")
train = ds["train_prefs"]
# Each example has: prompt, chosen, rejected, score_chosen, score_rejected
```

### Notes
- Primary dataset for DPO training experiments
- Used to train Zephyr-7B-beta
- Higher quality than Anthropic HH-RLHF (GPT-4 scored)

---

## Dataset 2: Anthropic HH-RLHF

### Overview
- **Source**: [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Size**: 160,800 train + 8,552 test examples (~70 MB)
- **Format**: HuggingFace Dataset (chosen/rejected conversation pairs)
- **Task**: RLHF preference alignment
- **Splits**: train (160,800), test (8,552)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf")
ds.save_to_disk("datasets/anthropic_hh_rlhf")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/anthropic_hh_rlhf")
# Each example has: chosen, rejected (full conversation strings)
```

### Notes
- The canonical RLHF preference dataset
- Used in most alignment tax papers (Lin et al., Kirk et al.)
- Contains both helpfulness and harmlessness preference data

---

## Dataset 3: TruthfulQA

### Overview
- **Source**: [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- **Size**: 817 questions across 38 categories
- **Format**: HuggingFace Dataset (questions with correct/incorrect answers)
- **Task**: Alignment tax evaluation (truthfulness)
- **Splits**: validation (817)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds.save_to_disk("datasets/truthfulqa")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/truthfulqa")
# Each example has: question, best_answer, correct_answers, incorrect_answers, category
```

### Notes
- Alignment should *improve* truthfulness — use this to verify alignment gains
- Small enough for quick evaluation
- Standard benchmark in alignment tax studies

---

## Dataset 4: WritingPrompts (Test Split)

### Overview
- **Source**: [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts)
- **Size**: 15,138 test examples (full dataset: 272,600 train)
- **Format**: HuggingFace Dataset (writing prompts + human stories)
- **Task**: Creative diversity measurement
- **Splits**: test (15,138)

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("euclaise/writingprompts", split="test")
ds.save_to_disk("datasets/writingprompts_test")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/writingprompts_test")
# Each example has: prompt, story
```

### Notes
- Use prompts to generate multiple completions from base vs aligned models
- Measure diversity metrics (distinct-n, Sent-BERT, entropy) on completions
- Creative writing domain makes style diversity highly visible

---

## Additional Datasets (Not Downloaded — Available on HuggingFace)

For completeness, these datasets are relevant but not pre-downloaded:

- **MMLU** (`cais/mmlu`): 14K multiple-choice questions for knowledge evaluation
- **HellaSwag** (`Rowan/hellaswag`): Commonsense reasoning benchmark
- **AlpacaFarm** (`tatsu-lab/alpaca_farm`): Instruction following with preferences
- **Blog Authorship Corpus** (`barilan/blog_authorship_corpus`): 681K blog posts for style analysis
- **Victorian-Era Authorship** (`NicholasSynovic/Victorian-Era-Authorship-Attribution`): Author attribution dataset
