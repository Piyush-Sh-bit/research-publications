import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from dataset import DocumentDataset, get_dataloader
from engine import BenchmarkEngine
from metrics import anls_score

def run_experiment(model_id, data_path, image_dir, batch_size, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model: {model_id} onto {device}...")
    
    # 1. Initialize HuggingFace Model & Processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, # Optimize memory
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    # 2. Setup Data and Engine
    dataloader = get_dataloader(data_path, image_dir, batch_size=batch_size)
    engine = BenchmarkEngine(device)
    
    results = []
    
    # 3. Warmup
    print("Preparing Warmup...")
    dummy_batch = next(iter(dataloader))
    def inference_wrapper(inputs):
        return model.generate(**inputs, max_new_tokens=64)
    # Warmup uses a minimal dummy pass (logic implementation depends on specific model API)
    
    # 4. Evaluation Loop
    print("Starting Main Evaluation Loop...")
    for iter_i, batch in enumerate(tqdm(dataloader)):
        images = batch["images"]
        questions = batch["questions"]
        gts = batch["gts"]
        
        # Prepare inputs (Varies greatly by model - Moondream vs Qwen vs InternVL)
        # Below is a standard LLaVA / Gen VLM approach:
        prompts = [f"<image>\n{q}" for q in questions]
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
        
        # Run through our hardware engine
        metrics = engine.run_inference_with_metrics(inference_wrapper, inputs, batch_size=len(images))
        
        # Decode texts
        generated_ids = metrics["outputs"]
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Log Document-level details
        for i in range(len(images)):
            pred_text = preds[i].split("Assistant:")[-1].strip() # Example split
            anls = anls_score(pred_text, gts[i])
            results.append({
                "model": model_id,
                "task": batch["tasks"][i],
                "script": batch["scripts"][i],
                "ground_truth": gts[i],
                "prediction": pred_text,
                "anls": anls,
                "latency_ms": metrics["latency_ms"] / len(images), # Approx per doc
                "vram_peak_gb": metrics["peak_vram_gb"]
            })
            
    # 5. Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 6. Print Summary
    print("\n--- EXPERIMENT SUMMARY ---")
    print(f"Mean ANLS (Efficacy): {df['anls'].mean():.3f}")
    print(f"Average Latency (ms): {df['latency_ms'].mean():.1f}")
    print(f"Peak VRAM used (GB): {df['vram_peak_gb'].max():.2f}")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HF model string (e.g. 'vikhyatk/moondream2')")
    parser.add_argument("--data_path", type=str, required=True, help="Path to annotations JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Where to save results")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    run_experiment(args.model_id, args.data_path, args.image_dir, args.batch_size, args.output_csv)
