import argparse
import numpy as np
import timeit
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM, RMSNorm
from cs336_basics.optimizer import AdamW
from cs336_systems.triton_rmsnorm import RMSNormTritonModule, RMSNormPyTorchModule

def profile_model(model_config, context_length=128, vocab_size=10000, batch_size=16, 
                 num_warmup=1, num_measurements=5, device=None, use_mixed_precision=False,
                 passes_to_run=None):
    """
    Profile a model configuration using PyTorch profiler.
    
    Args:
        model_config: dict containing model parameters (d_model, num_layers, num_heads, d_ff)
        context_length: length of input sequences
        vocab_size: size of vocabulary
        batch_size: number of sequences to process at once
        num_warmup: number of warmup iterations before profiling
        num_measurements: number of profiled iterations
        device: device to run the model on
        use_mixed_precision: whether to use mixed precision (FP16)
        passes_to_run: list of passes to run ('forward', 'backward', 'optimizer')
    
    Returns:
        None (results are printed and exported to a file)
    """
    # Defaults
    if passes_to_run is None:
        passes_to_run = ['forward', 'backward', 'optimizer']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = nullcontext
    
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        attn_pdrop=model_config.get('attn_pdrop'),
        residual_pdrop=model_config.get('residual_pdrop')
    )
    if  model_config.get('compile_model', False):
        model = torch.compile(model)
    model = model.to(device)
    
    # Create random input data and targets
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Warmup iterations
    warmup_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    for _ in range(num_warmup):
        with amp_context():
            outputs = model(warmup_data)
            
            # Only compute loss and backward if running those passes
            if 'backward' in passes_to_run:
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Only step optimizer if running optimizer pass
                if 'optimizer' in passes_to_run:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    precision_label = 'mixed' if use_mixed_precision else 'full'
    passes_label = '_'.join(passes_to_run)
    print(f"Running profiler for {model_config['name']} model with {precision_label} precision ({passes_label} passes)...")
    
    def run_step(model, inputs, targets, optimizer, criterion):
        if 'forward' in passes_to_run:
            with record_function("forward_pass"):
                with amp_context():
                    outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            with amp_context():
                outputs = model(inputs)
        
        if 'backward' in passes_to_run:
            with record_function("backward_pass"):
                with amp_context():
                    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
            if 'optimizer' in passes_to_run:
                with record_function("optimizer"):
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
    try:
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        ) as prof:
            for _ in range(num_measurements):
                run_step(model, input_ids, targets, optimizer, criterion)
                prof.step()
        profiler_output_name = f"{model_config['name']}_{precision_label}_{passes_label}_profiler_stacks.txt"
        prof.export_stacks(profiler_output_name, "self_cuda_time_total")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    except Exception as e:
        print(f"Profiler error: {str(e)}")


def profile_memory(model_config, context_length=128, vocab_size=10000, batch_size=16, 
                  num_warmup=1, num_measurements=3, device=None, use_mixed_precision=False,
                  passes_to_run=None):
    """
    Profile memory usage of a model configuration using PyTorch memory profiler.
    
    Args:
        model_config: dict containing model parameters (d_model, num_layers, num_heads, d_ff)
        context_length: length of input sequences
        vocab_size: size of vocabulary
        batch_size: number of sequences to process at once
        num_warmup: number of warmup iterations before profiling
        num_measurements: number of profiled iterations
        device: device to run the model on
        use_mixed_precision: whether to use mixed precision (FP16)
        passes_to_run: list of passes to run ('forward', 'backward', 'optimizer')
    
    Returns:
        None (results are exported to files)
    """
    # Defaults
    if passes_to_run is None:
        passes_to_run = ['forward', 'backward', 'optimizer']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = nullcontext
    
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        attn_pdrop=model_config.get('attn_pdrop'),
        residual_pdrop=model_config.get('residual_pdrop')
    )
    if model_config.get('compile_model', False):
        model = torch.compile(model)
    model = model.to(device)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    warmup_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    for _ in range(num_warmup):
        with amp_context():
            outputs = model(warmup_data)
            
            if 'backward' in passes_to_run:
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if 'optimizer' in passes_to_run:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    precision_label = 'mixed' if use_mixed_precision else 'full'
    passes_label = '_'.join(passes_to_run)
    print(f"Running memory profiler for {model_config['name']} model with {precision_label} precision ({passes_label} passes)...")
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    n_steps = num_measurements
    
    def run_step(model, inputs, targets, optimizer, criterion):
        if 'forward' in passes_to_run:
            with record_function("forward_pass"):
                with amp_context():
                    outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            with amp_context():
                outputs = model(inputs)
        
        if 'backward' in passes_to_run:
            with record_function("backward_pass"):
                with amp_context():
                    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
            if 'optimizer' in passes_to_run:
                with record_function("optimizer"):
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    try:
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
            experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(n_steps):
                run_step(model, input_ids, targets, optimizer, criterion)
                prof.step()
            
            # Save a graphical timeline of memory usage
            # timeline_file = f"{model_config['name']}_{precision_label}_{passes_label}_timeline.html"
            # prof.export_memory_timeline(timeline_file, device=device)
            
        # Save a pickle file to be loaded by PyTorch's online tool
        snapshot_file = f"{model_config['name']}_{precision_label}_{passes_label}_memory_snapshot.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_file)
        
        # Stop recording history
        torch.cuda.memory._record_memory_history(enabled=None)
                    
    except Exception as e:
        print(f"Memory profiler error: {str(e)}")
        torch.cuda.memory._record_memory_history(enabled=None)


def benchmark_model(model_config, context_length=128, vocab_size=10000, batch_size=16, 
                   num_warmup=1, num_measurements=5, device=None, use_mixed_precision=False,
                   passes_to_run=None):
    """
    Benchmark a model configuration by timing forward and backward passes using timeit.
    
    Args:
        model_config: dict containing model parameters (d_model, num_layers, num_heads, d_ff)
        context_length: length of input sequences
        vocab_size: size of vocabulary
        batch_size: number of sequences to process at once
        num_warmup: number of warmup iterations before timing
        num_measurements: number of timed iterations to average over
        device: device to run the model on
        use_mixed_precision: whether to use mixed precision (FP16)
        passes_to_run: list of passes to run ('forward', 'backward', 'optimizer')
    
    Returns:
        Dictionary with timing statistics for requested passes
    """
    if passes_to_run is None:
        passes_to_run = ['forward', 'backward', 'optimizer']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up mixed precision context
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = nullcontext
    
    # Create model with the given configuration
    norms = {
        'rms': RMSNorm,
        'layer': nn.LayerNorm,
        'triton': RMSNormTritonModule,
        'pytorch': RMSNormPyTorchModule,
    }
    norm_func = norms[model_config.get('norm_type', 'rms')]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        attn_pdrop=model_config.get('attn_pdrop'),
        residual_pdrop=model_config.get('residual_pdrop'),
        norm_layer=norm_func,
        compile_norm_layer=model_config.get('compile_model', False),
    )

    model = model.to(device)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    warmup_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    for _ in range(num_warmup):
        with amp_context():
            outputs = model(warmup_data)
            
            if 'backward' in passes_to_run:
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if 'optimizer' in passes_to_run:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    def run_and_time_passes():
        times = {}
        
        torch.cuda.synchronize() 
        start = timeit.default_timer()
        
        with amp_context():
            outputs = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if 'forward' in passes_to_run:
            times['forward'] = timeit.default_timer() - start

        if 'backward' in passes_to_run:
            start = timeit.default_timer()
            
            with amp_context():
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            times['backward'] = timeit.default_timer() - start

        if 'optimizer' in passes_to_run:
            start = timeit.default_timer()
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            times['optimizer'] = timeit.default_timer() - start

        return times


    all_times = {pass_name: [] for pass_name in passes_to_run}
    
    for _ in range(num_measurements):
        times = run_and_time_passes()
        for pass_name, time_value in times.items():
            all_times[pass_name].append(time_value)
    
    results = {'precision': 'mixed' if use_mixed_precision else 'full'}
    
    for pass_name in passes_to_run:
        if pass_name in all_times and all_times[pass_name]:
            results[f'{pass_name}_mean'] = np.mean(all_times[pass_name])
            results[f'{pass_name}_std'] = np.std(all_times[pass_name])
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark transformer model configurations using timeit')
    
    # Model configuration parameters
    parser.add_argument('--model-sizes', type=str, default='all', 
                        help='Which model sizes to benchmark (comma-separated list from: S,M,L,XL or "all")')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Override model dimension for custom benchmarking')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override number of layers for custom benchmarking')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Override number of attention heads for custom benchmarking')
    parser.add_argument('--d-ff', type=int, default=None,
                        help='Override feed-forward dimension for custom benchmarking')
    parser.add_argument('--attn-pdrop', type=float, default=None,
                        help='Dropout rate for attention probabilities')
    parser.add_argument('--residual-pdrop', type=float, default=None,
                        help='Dropout rate for residual connections')
    parser.add_argument('--norm-type', type=str, default='rms', choices=['rms', 'layer', 'triton', 'pytorch'],
                    help='Normalization type to use: rms (default) or layer')
    parser.add_argument('--compile-model', action='store_true', help='Apply torch.compile to model')
    
    # New parameter for selecting passes to run
    parser.add_argument('--passes', type=str, default='forward,backward,optimizer',
                        help='Comma-separated list of passes to run (forward,backward,optimizer)')
    
    # Optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for the AdamW optimizer')
    
    # Benchmarking parameters
    parser.add_argument('--context-length', type=int, default=128,
                        help='Context length for the model')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size for the model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for benchmarking')
    parser.add_argument('--num-warmup', type=int, default=1,
                        help='Number of warmup iterations before timing')
    parser.add_argument('--num-measurements', type=int, default=5,
                        help='Number of timed iterations to average over')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda, cpu, or specific cuda device like cuda:0)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Mixed precision option
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision (FP16) for the model')
    parser.add_argument('--compare-precision', action='store_true',
                        help='Compare full vs. mixed precision for each model size')
    
    # PyTorch profiler option
    parser.add_argument('--use-profiler', action='store_true',
                        help='Enable PyTorch profiler for detailed performance analysis')
    
    # Memory profiler option (new)
    parser.add_argument('--profile-memory', action='store_true',
                        help='Enable PyTorch memory profiler to analyze memory usage')
    
    # Output configuration
    parser.add_argument('--csv', type=str, default=None,
                        help='Save results to CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse passes to run
    passes_to_run = [p.strip() for p in args.passes.split(',') if p.strip()]
    valid_passes = ['forward', 'backward', 'optimizer']
    
    # Validate passes
    for p in passes_to_run:
        if p not in valid_passes:
            print(f"Warning: Invalid pass '{p}', must be one of {valid_passes}")
            passes_to_run.remove(p)
    
    if not passes_to_run:
        print("No valid passes specified, using default (forward,backward,optimizer)")
        passes_to_run = valid_passes
    
    print(f"Running passes: {', '.join(passes_to_run)}")
    
    # Set device based on arguments
    if args.no_cuda:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if (args.use_profiler or args.profile_memory) and not torch.cuda.is_available():
        print("Warning: PyTorch profiler with CUDA features requested, but CUDA is not available.")
        print("Falling back to CPU-only profiling.")
    
    # Check if mixed precision is available
    if (args.mixed_precision or args.compare_precision) and not torch.cuda.is_available():
        print("Warning: Mixed precision requested, but CUDA is not available.")
        print("Mixed precision will be disabled.")
        args.mixed_precision = False
        args.compare_precision = False
    
    standard_configs = {
        'S': {'name': 'S', 'd_model': 768, 'num_layers': 12, 'num_heads': 12, 'd_ff': 3072},
        'M': {'name': 'M', 'd_model': 1024, 'num_layers': 24, 'num_heads': 16, 'd_ff': 4096},
        'L': {'name': 'L', 'd_model': 1280, 'num_layers': 36, 'num_heads': 20, 'd_ff': 5120},
        # 'XL': {'name': 'XL', 'd_model': 1600, 'num_layers': 48, 'num_heads': 25, 'd_ff': 6400},
    }
    
    if args.model_sizes.lower() == 'all':
        model_configs = list(standard_configs.values())
    else:
        model_configs = [standard_configs[size] for size in args.model_sizes.split(',') if size in standard_configs]
    
    # Add custom configuration if parameters are provided
    if any(param is not None for param in [args.d_model, args.num_layers, args.num_heads, args.d_ff]):
        custom_config = {'name': 'Custom'}
        if args.d_model is not None:
            custom_config['d_model'] = args.d_model
        else:
            custom_config['d_model'] = 512
            
        if args.num_layers is not None:
            custom_config['num_layers'] = args.num_layers
        else:
            custom_config['num_layers'] = 4
            
        if args.num_heads is not None:
            custom_config['num_heads'] = args.num_heads
        else:
            custom_config['num_heads'] = 8
            
        if args.d_ff is not None:
            custom_config['d_ff'] = args.d_ff
        else:
            custom_config['d_ff'] = 2048
            
        if args.attn_pdrop is not None:
            custom_config['attn_pdrop'] = args.attn_pdrop
            
        if args.residual_pdrop is not None:
            custom_config['residual_pdrop'] = args.residual_pdrop
            
        model_configs = [custom_config]
    
    if args.attn_pdrop is not None or args.residual_pdrop is not None:
        for config in model_configs:
            if args.attn_pdrop is not None:
                config['attn_pdrop'] = args.attn_pdrop
            if args.residual_pdrop is not None:
                config['residual_pdrop'] = args.residual_pdrop
    
    for config in model_configs:
        config['compile_model'] = args.compile_model
        config['norm_type'] = args.norm_type
        config['name'] += f"_{args.norm_type}"

    print(f"Running on {device}")
    print(f"Context length: {args.context_length}, Vocab size: {args.vocab_size}, Batch size: {args.batch_size}")
    print(f"Warmup steps: {args.num_warmup}, Measurement steps: {args.num_measurements}")
    print(f"Using AdamW optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}")
    print(f"Compiled model?:  {args.compile_model}")

    if args.mixed_precision:
        print("Using mixed precision (FP16)")
    elif args.compare_precision:
        print("Comparing full precision (FP32) vs. mixed precision (FP16)")
    
    if args.profile_memory and args.use_profiler:
        print("Both memory profiling and performance profiling are enabled.")
        print("Running both profilers sequentially...")
    
    if args.profile_memory:
        print("Running PyTorch memory profiler - memory usage data will be collected")
        print("-" * 80)
        
        precision_modes = [False, True] if args.compare_precision else [args.mixed_precision]
        
        for config in model_configs:
            for use_mixed_precision in precision_modes:
                profile_memory(
                    config,
                    context_length=args.context_length,
                    vocab_size=args.vocab_size,
                    batch_size=args.batch_size,
                    num_warmup=args.num_warmup,
                    num_measurements=args.num_measurements,
                    device=device,
                    use_mixed_precision=use_mixed_precision,
                    passes_to_run=passes_to_run
                )
                print("-" * 80)
    
    if args.use_profiler:
        print("Running PyTorch profiler - detailed performance data will be collected")
        print("-" * 80)
        
        precision_modes = [False, True] if args.compare_precision else [args.mixed_precision]
        
        for config in model_configs:
            for use_mixed_precision in precision_modes:
                profile_model(
                    config,
                    context_length=args.context_length,
                    vocab_size=args.vocab_size,
                    batch_size=args.batch_size,
                    num_warmup=args.num_warmup,
                    num_measurements=args.num_measurements,
                    device=device,
                    use_mixed_precision=use_mixed_precision,
                    passes_to_run=passes_to_run
                )
                print("-" * 80)
    
    if not args.use_profiler and not args.profile_memory:
        if args.compare_precision:
            precision_modes = [False, True]
            header_width = 150
            print("Comparing full precision (FP32) vs. mixed precision (FP16)")
            print("-" * header_width)
            
            header = f"{'Model':^10} | {'d_model':^8} | {'Layers':^8} | {'Heads':^8} | {'d_ff':^8} | {'Precision':^10}"
            
            for pass_name in passes_to_run:
                header += f" | {f'{pass_name.capitalize()} Time (ms)':^15} | {f'{pass_name.capitalize()} Std (ms)':^15}"
            
            header += f" | {'Speedup':^8}"
            
            print(header)
            print("-" * header_width)
        else:
            precision_modes = [args.mixed_precision]
            header_width = 130
            print("Benchmarking models")
            print("-" * header_width)
            
            header = f"{'Model':^10} | {'d_model':^8} | {'Layers':^8} | {'Heads':^8} | {'d_ff':^8}"
            
            for pass_name in passes_to_run:
                header += f" | {f'{pass_name.capitalize()} Time (ms)':^15} | {f'{pass_name.capitalize()} Std (ms)':^15}"
            
            print(header)
            print("-" * header_width)
        
        results = []
        
        for config in model_configs:
            config_results = {}
            
            for use_mixed_precision in precision_modes:
                result = benchmark_model(
                    config, 
                    context_length=args.context_length,
                    vocab_size=args.vocab_size,
                    batch_size=args.batch_size,
                    num_warmup=args.num_warmup,
                    num_measurements=args.num_measurements,
                    device=device,
                    use_mixed_precision=use_mixed_precision,
                    passes_to_run=passes_to_run
                )
                
                # Add config info to results
                result.update(config)
                results.append(result)
                
                # Store results for comparison
                if args.compare_precision:
                    key = f"{config['name']}_{'mixed' if use_mixed_precision else 'full'}"
                    config_results[key] = result
                
                # Build output line with only the requested pass data
                precision_label = "Mixed" if use_mixed_precision else "Full"
                if args.compare_precision:
                    output_line = f"{config['name']:^10} | {config['d_model']:^8d} | {config['num_layers']:^8d} | " + \
                                  f"{config['num_heads']:^8d} | {config['d_ff']:^8d} | {precision_label:^10}"
                else:
                    output_line = f"{config['name']:^10} | {config['d_model']:^8d} | {config['num_layers']:^8d} | " + \
                                  f"{config['num_heads']:^8d} | {config['d_ff']:^8d}"
                
                # Add timing data for each pass
                for pass_name in passes_to_run:
                    if f'{pass_name}_mean' in result:
                        pass_time_ms = result[f'{pass_name}_mean'] * 1000
                        pass_std_ms = result[f'{pass_name}_std'] * 1000
                        output_line += f" | {pass_time_ms:^15.2f} | {pass_std_ms:^15.2f}"
                    else:
                        output_line += f" | {'N/A':^15} | {'N/A':^15}"
                
                if args.compare_precision:
                    speedup = ""
                    
                    if f"{config['name']}_full" in config_results and f"{config['name']}_mixed" in config_results:
                        full_result = config_results[f"{config['name']}_full"]
                        mixed_result = config_results[f"{config['name']}_mixed"]
                        
                        if use_mixed_precision:
                            full_total = sum([full_result.get(f'{p}_mean', 0) for p in passes_to_run])
                            mixed_total = sum([mixed_result.get(f'{p}_mean', 0) for p in passes_to_run])
                            
                            if mixed_total > 0:
                                speedup = f"{full_total / mixed_total:.2f}x"
                    
                    output_line += f" | {speedup:^8}"
                
                print(output_line)
        
        print("-" * (header_width if args.compare_precision else header_width))
        
        if results and args.compare_precision:
            largest_model = None
            for config in model_configs:
                model_size = config['d_model'] * config['num_layers']
                if largest_model is None or model_size > largest_model['size']:
                    largest_model = {
                        'name': config['name'],
                        'size': model_size,
                        'config': config
                    }
            
            # Get results for the largest model
            largest_full = None
            largest_mixed = None
            
            for result in results:
                if result['name'] == largest_model['name']:
                    if result['precision'] == 'full':
                        largest_full = result
                    else:
                        largest_mixed = result
            
            if largest_full and largest_mixed:
                print("\nComparison for largest model:")
                
                total_full = 0
                total_mixed = 0
                
                for pass_name in passes_to_run:
                    if f'{pass_name}_mean' in largest_full and f'{pass_name}_mean' in largest_mixed:
                        full_ms = largest_full[f'{pass_name}_mean'] * 1000
                        mixed_ms = largest_mixed[f'{pass_name}_mean'] * 1000
                        
                        speedup = full_ms / mixed_ms if mixed_ms > 0 else 0
                        print(f"  {pass_name.capitalize()}: {speedup:.2f}x speedup")
                        
                        total_full += largest_full[f'{pass_name}_mean']
                        total_mixed += largest_mixed[f'{pass_name}_mean']
                
                if total_mixed > 0:
                    total_speedup = total_full / total_mixed
                    print(f"  Total: {total_speedup:.2f}x speedup with mixed precision")
                
                if len(model_configs) > 1:
                    speedups = []
                    for config in model_configs:
                        full_result = None
                        mixed_result = None
                        
                        for result in results:
                            if result['name'] == config['name']:
                                if result['precision'] == 'full':
                                    full_result = result
                                else:
                                    mixed_result = result
                        
                        if full_result and mixed_result:
                            full_total = sum([full_result.get(f'{p}_mean', 0) for p in passes_to_run])
                            mixed_total = sum([mixed_result.get(f'{p}_mean', 0) for p in passes_to_run])
                            
                            if mixed_total > 0:
                                speedup = full_total / mixed_total
                                speedups.append((config['name'], speedup))
                    
                    if speedups:
                        print("\nSpeedup trends by model size:")
                        for name, speedup in speedups:
                            print(f"  {name}: {speedup:.2f}x")
        
        if args.csv:
            import csv
            with open(args.csv, 'w', newline='') as csvfile:
                fieldnames = ['name', 'd_model', 'num_layers', 'num_heads', 'd_ff', 'precision']
                
                for pass_name in passes_to_run:
                    fieldnames.extend([f'{pass_name}_mean', f'{pass_name}_std'])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow({k: result[k] for k in fieldnames if k in result})

if __name__ == "__main__":
    main()