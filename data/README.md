# data/prompts_96.jsonl

96 prompts (one JSON object per line: `id`, `source`, `domain`, `thinking`, `max_tokens`, `prompt`) taken from seven
public datasets. This is the subset listed in `scripts/subset_96.txt`; `scripts/build_prompts.py --subset
scripts/subset_96.txt` regenerates exactly this file.

| source | domain | count | dataset | license |
|---|---|---|---|---|
| gsm8k | math | 19 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (main, train) | MIT |
| math500 | math | 16 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (test); from [openai/prm800k](https://github.com/openai/prm800k) / [hendrycks/math](https://github.com/hendrycks/math) | MIT |
| mbpp | code | 17 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (full, train) | CC BY 4.0 |
| humaneval | code | 7 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (test) | MIT |
| ultrachat | chat | 19 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (train_sft) | MIT |
| openorca | chat | 11 | [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) (train) | MIT |
| zh_alpaca | zh | 7 | [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) (train) | CC BY 4.0 |

Changes made to the original records (see `scripts/build_prompts.py`):
- **math500**: appended "Please reason step by step."
- **mbpp**: the task text is followed by "Write a Python function and explain it. Example test: " and the record's
  first test case.
- **humaneval**: the function stub is preceded by "Complete the following Python function and explain your solution."
- **openorca**: the system prompt, if any, is prepended to the question.
- **zh_alpaca**: the instruction and input are joined with a newline.
- **All sources**: surrounding whitespace is stripped.

Each prompt remains under its source dataset's license: MIT, or CC BY 4.0 for mbpp and zh_alpaca, which requires
attribution to the original authors (linked above). The MIT license of this repository covers the code only.
