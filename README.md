# Ollama Finetune Setup for Bielik Model

This setup allows you to finetune the Bielik-4.5B-v3.0-Instruct model using Ollama and optionally llama.cpp for LoRA finetuning.

## Prerequisites

- Windows 10/11
- Python 3.8+
- Git
- CMake
- Ollama installed
- CUDA toolkit (for GPU support in llama.cpp)

## Folder Structure

- `data/`: Dataset files (sft_dataset.jsonl, train.jsonl, val.jsonl, train.txt)
- `models/`: Model files (base.gguf, lora-ssomar.gguf)
- `scripts/`: Python scripts (validate_split.py, make_train.py, autotest.py)
- `tools/`: llama.cpp repository
- `logs/`: Log files

## Usage

Run `run.bat` and follow the menu:

1. **Train model**: Validates dataset, splits, creates train.txt, pulls base model, runs finetune, creates Ollama model with adapter.
2. **Test model**: Runs autotest on the finetuned model.
3. **Update dataset**: Validates and splits the dataset.
4. **Rebuild llama.cpp**: Clones and builds llama.cpp with CUDA support.
5. **Scrape and create dataset**: Scrapes websites and generates sft_dataset.jsonl.
6. **Reduce dataset**: Reduces the number of lines in sft_dataset.jsonl (e.g., from 1200 to 800).

## Building llama.cpp

To enable GPU support for finetuning:

1. Install CUDA toolkit from NVIDIA.
2. In the BUILD option, it runs `cmake .. -DLLAMA_CUBLAS=ON` to enable CUDA.
3. Build with `cmake --build . --config Release`.

If you encounter issues, ensure CUDA is properly installed and the GPU drivers are up to date.

## Dataset Format

`sft_dataset.jsonl` should contain JSONL with each line as:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]}
```

## Notes

- The setup downloads the base GGUF from HuggingFace.
- Finetuning uses LoRA with r=8, alpha=16.
- The Ollama model is created with the LoRA adapter.
- All scripts handle UTF-8 and log to logs/.
