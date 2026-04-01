import torch
import time
from typing import Callable, Any, Dict, Tuple

class BenchmarkEngine:
    """
    Engine to measure end-to-end generation latency, peak VRAM, and throughput.
    """
    def __init__(self, device="cuda"):
        self.device = device
        if "cuda" not in self.device:
            print("WARNING: Hardware benchmarking requires a CUDA device.")

    def run_inference_with_metrics(self, model: Callable, inputs: Any, batch_size: int = 1) -> Dict:
        """
        Runs a single batch and captures end-to-end generation latency and peak VRAM.
        For true TTFT, generation must support token-level streaming instrumentation.
        """
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # --- MODEL INFERENCE ---
        with torch.no_grad():
            outputs = model(inputs)
        # -----------------------
        
        end_event.record()
        torch.cuda.synchronize()
        
        latency_ms = start_event.elapsed_time(end_event)
        
        # Capture Peak VRAM in GB
        peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
        
        return {
            "outputs": outputs,
            "latency_ms": latency_ms,
            "peak_vram_gb": peak_vram_gb,
            "tps": (batch_size * 1000) / latency_ms  # Documents per second
        }

    def warmup(self, model: Callable, inputs: Any, iterations: int = 5):
        """Warm up the GPU to ensure stable clock speeds."""
        print(f"Warming up model for {iterations} iterations...")
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(inputs)
        torch.cuda.synchronize()
        print("Warmup complete.")
