# DeepTweak Experiment Visualization

This project analyzes the impact of OS kernel parameter tweaks on AI workload performance. The visualization script processes benchmark data from multiple experiments and generates insightful visualizations.

## Directory Structure

- `data/` - Contains the experiment data
  - `base/` - Baseline experiment (no tweaks)
  - `exp1/`, `exp2/`, `exp3/` - Different OS parameter tweak experiments
  - Each experiment folder contains:
    - `benchmark.csv` - Performance metrics collected during the experiment
    - `skyhook.yaml` - Configuration of OS kernel parameters applied

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the visualization script:

```bash
python visualize.py
```

This will:
1. Load all experiment data from the `data/` directory
2. Calculate statistics for each experiment
3. Generate visualizations comparing the experiments
4. Create a summary table of results

## Output

The script generates the following outputs:

- `plots/` directory:
  - Time series plots comparing metrics over time
  - Bar charts comparing average metrics across experiments
  - Kernel parameter impact visualization
  - Performance improvement comparison

- `results/` directory:
  - `experiment_summary.csv` - Summary table with key metrics for each experiment

## Metrics Analyzed

The script analyzes the following metrics:
- Step time
- Samples per second
- CPU utilization
- Memory utilization
- GPU utilization
- GPU power consumption
- GPU temperature
- GPU memory utilization

## Interpreting Results

The visualizations help identify:
1. Which kernel parameter tweaks resulted in performance improvements
2. The impact of different tweaks on various system metrics
3. Trade-offs between performance and resource utilization
