import torch
from fvcore.nn import FlopCountAnalysis


def profile(model, *inputs, num_warmup=10, num_measure=50):
    """Profiles a PyTorch module for parameters, FLOPs, inference latency, and peak CUDA memory.

    Args:
        model: The PyTorch nn.Module to profile.
        *inputs: Tensors to pass into the model.
        num_warmup: Number of forward passes to run before measuring latency.
        num_measure: Number of forward passes to run for latency averaging.

    Returns:
        A dictionary containing the profiling statistics.
    """
    model = model.cuda()
    model.eval()

    # move all provided inputs to the target device
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    # 1. parameter count
    stats = {"parameters": sum(p.numel() for p in model.parameters())}

    # 2. FLOPs calculation (fvcore)
    flops_analyzer = FlopCountAnalysis(model, inputs)
    flops_analyzer.unsupported_ops_warnings(False)
    stats["flops"] = flops_analyzer.total() * 2

    # 3. peak CUDA memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda")
    with torch.no_grad():
        _ = model(*inputs)
    stats["peak_memory_mb"] = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)

    # 4. inference latency
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*inputs)
    torch.cuda.synchronize("cuda")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
    with torch.no_grad():
        for i in range(num_measure):
            start_events[i].record()
            _ = model(*inputs)
            end_events[i].record()
    torch.cuda.synchronize("cuda")
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    stats["latency_mean_ms"] = sum(times_ms) / len(times_ms)
    stats["latency_min_ms"] = min(times_ms)
    stats["latency_max_ms"] = max(times_ms)

    return stats
