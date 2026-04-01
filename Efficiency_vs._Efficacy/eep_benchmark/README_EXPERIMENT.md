# Efficiency-Efficacy Pareto (EEP) Benchmark

This package contains the PyTorch/HuggingFace codebase to run physical inference benchmarks for your EEP workflow. Running this code on a GPU generates a `results.csv` file with real latency, throughput, and VRAM measurements that can be integrated into your notebook analysis.

## Prerequisites
Run this on an environment with a dedicated CUDA GPU (e.g., RunPod, AWS EC2, local Server).
```bash
pip install torch transformers pillow pandas tqdm python-Levenshtein accelerate
```

## Structure
- `dataset.py`: Handles loading your 18,420 Scanned Images Dataset (SID) alongside their JSON annotations.
- `engine.py`: A `torch.cuda` hardware engine that measures end-to-end generation latency and tracks peak VRAM using `Event` and `max_memory_allocated`.
- `metrics.py`: The exact mathematical implementation of your ANLS text comparison and EEP score generators.
- `run_inference.py`: Evaluates a specific model from HuggingFace.

## How to run an experiment

1. Place your dataset inside a folder.
2. Structure your truth pairs in a `data.jsonl` file. Format:
   `{"image_path": "forms/001.jpg", "question": "Total Value?", "answer": "$150", "task": "Invoice", "script": "Latin"}`
3. Run the Python script for the model of your choice:

```bash
python run_inference.py \
    --model_id "vikhyatk/moondream2" \
    --data_path "path/to/data.jsonl" \
    --image_dir "path/to/images/" \
    --output_csv "results_moondream2_seed1.csv" \
    --batch_size 1
```

Once the loop completes, it will generate a `results_moondream2_seed1.csv` file. 

Take this CSV file back to your Jupyter Notebook (`code.ipynb`), and replace the `generate_experiment_run` function (the one generating synthetic distributions) with `pd.read_csv("results_moondream2_seed1.csv")`.

This allows you to calibrate simulation assumptions with real hardware measurements and report deployment-specific EEP values transparently.