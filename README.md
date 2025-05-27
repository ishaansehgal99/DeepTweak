# DeepTweak

**Performance analysis tool for AI workloads with OS kernel parameter optimization**

DeepTweak analyzes how different Linux kernel parameter tweaks affect AI training performance on Kubernetes. This project uses [KAITO](https://github.com/Azure/kaito) for AI workload orchestration and [Skyhook](https://github.com/skyhookdm/skyhook) for OS-level optimizations, all running on Kubernetes infrastructure.

## What it does

- Compares AI training performance across different kernel parameter configurations
- Analyzes metrics like throughput, GPU utilization, memory usage, and training speed
- Generates visualizations showing the impact of specific kernel tweaks
- Identifies which optimizations provide the best performance improvements
- Demonstrates cloud-native AI optimization using Kubernetes ecosystem tools

## Architecture

- **KAITO** - Manages AI model inference and training workloads on Kubernetes
- **Skyhook** - Applies OS kernel parameter optimizations via Kubernetes DaemonSets
- **Kubernetes** - Orchestrates the entire experimental infrastructure
- **DeepTweak** - Analyzes and visualizes the performance impact

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis:**
   ```bash
   python visualize.py
   ```

3. **View results:**
   - Plots saved to `plots_*/` directories
   - Summary data in `results_*/` directories

## Data Structure

```
data/
├── one_node_one_gpu/          # Single GPU experiments
│   ├── base/                  # Baseline (no tweaks)
│   ├── exp1/, exp2/, exp3/    # Different kernel configurations
├── one_node_two_gpu/          # Multi-GPU experiments
    └── ...
```

Each experiment contains:
- `benchmark.csv` - Performance metrics over time
- `skyhook.yaml` - Kernel parameter configuration applied via Skyhook

## Key Metrics

- **Step time** - Training iteration duration
- **Samples/sec** - Training throughput
- **GPU utilization** - Hardware efficiency
- **Memory usage** - Resource consumption
- **Temperature & Power** - Thermal/energy impact

## Output

The tool generates:
- Time series plots comparing performance over time
- Bar charts showing average improvements
- Kernel parameter impact analysis
- Performance summary tables

Perfect for identifying which kernel tweaks actually improve AI training performance in your environment.
