import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from cs336_basics.model import RMSNorm
from cs336_systems.triton_rmsnorm import RMSNormTritonModule, RMSNormPyTorchModule


def benchmark_layer(norm_func, dim, device, runs=1000, trials=5, run_backward=False, compile=False):
    x = torch.randn(50000, dim, device=device)
    if compile:
        layer = torch.compile(norm_func(dim)).to(device)
    else:
        layer = norm_func(dim).to(device)
    
    if run_backward:
        dy = torch.randn(50000, dim, device=device)
    
    forward_trial_times = []
    backward_trial_times = []
    
    for _ in range(trials):
        # Warm-up
        for _ in range(10):
            # Clear gradients
            if run_backward:
                x.grad = None
                x.requires_grad_(True)
            else:
                x.requires_grad_(False)
                
            _ = layer(x)
            
            if run_backward:
                _.backward(dy)
                
        torch.cuda.synchronize()

        forward_times = []
        backward_times = []
        
        for _ in range(runs):
            # Clear gradients
            if run_backward:
                x.grad = None
                x.requires_grad_(True)
            else:
                x.requires_grad_(False)
            
            torch.cuda.synchronize()
            forward_start = time.time()
            result = layer(x)
            torch.cuda.synchronize()
            forward_end = time.time()
            forward_times.append((forward_end - forward_start) * 1000)
            
            if run_backward:
                torch.cuda.synchronize()
                backward_start = time.time()
                result.backward(dy)
                torch.cuda.synchronize()
                backward_end = time.time()
                backward_times.append((backward_end - backward_start) * 1000)
        
        forward_trial_times.append(np.mean(forward_times))
        if run_backward:
            backward_trial_times.append(np.mean(backward_times))

    forward_avg = np.mean(forward_trial_times)
    forward_std = np.std(forward_trial_times)
    
    if run_backward:
        backward_avg = np.mean(backward_trial_times)
        backward_std = np.std(backward_trial_times)
        combined_avg = forward_avg + backward_avg
        return forward_avg, forward_std, backward_avg, backward_std, combined_avg
    else:
        return forward_avg, forward_std, None, None, forward_avg

def main():
    parser = argparse.ArgumentParser(description='Benchmark normalization layers')
    parser.add_argument('--backward', action='store_true', 
                      help='Run backward pass in addition to forward pass')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = [1024, 2048, 4096]
    
    if args.backward:
        header = (f"{'Dim':>6} | {'Type':>15} | {'Forward (ms)':>12} | {'Backward (ms)':>12} | "
                  f"{'Combined (ms)':>12} | {'Fwd Std':>8} | {'Bwd Std':>8}")
    else:
        header = f"{'Dim':>6} | {'Type':>15} | {'Forward (ms)':>12} | {'Std Dev (ms)':>12}"
    
    print(header)
    print("-" * len(header))
    
    for dim in dims:
        rms_results = benchmark_layer(RMSNorm, dim, device, run_backward=args.backward)
        ln_results = benchmark_layer(nn.LayerNorm, dim, device, run_backward=args.backward)
        triton_results = benchmark_layer(RMSNormTritonModule, dim, device, run_backward=args.backward)
        compiled_pytorch_results = benchmark_layer(RMSNormPyTorchModule, dim, device, run_backward=args.backward, compile=True)

        if args.backward:
            for name, results in [("RMSNorm", rms_results), 
                                 ("LayerNorm", ln_results), 
                                 ("Triton RMSNorm", triton_results),
                                 ("Compiled RMSNorm", compiled_pytorch_results)]:
                fwd_avg, fwd_std, bwd_avg, bwd_std, combined = results
                print(f"{dim:6} | {name:>15} | {fwd_avg:12.4f} | {bwd_avg:12.4f} | "
                      f"{combined:12.4f} | {fwd_std:8.4f} | {bwd_std:8.4f}")
        else:
            for name, results in [("RMSNorm", rms_results), 
                                 ("LayerNorm", ln_results), 
                                 ("Triton RMSNorm", triton_results),
                                 ("Compiled RMSNorm", compiled_pytorch_results)]:
                fwd_avg, fwd_std, _, _, _ = results
                print(f"{dim:6} | {name:>15} | {fwd_avg:12.4f} | {fwd_std:12.4f}")
                
        print("-" * len(header))

if __name__ == "__main__":
    main()