#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

# Set the style for our plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_experiment_data(base_dir='data'):
    """
    Load all experiment data from the data directory
    """
    experiments = {}
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for exp_dir in exp_dirs:
        exp_path = os.path.join(base_dir, exp_dir)
        
        # Load benchmark data
        benchmark_file = os.path.join(exp_path, 'benchmark.csv')
        if os.path.exists(benchmark_file):
            df = pd.read_csv(benchmark_file)
            
            # Load skyhook.yaml to understand the kernel tweaks
            skyhook_file = os.path.join(exp_path, 'skyhook.yaml')
            kernel_tweaks = "No tweaks"
            if os.path.exists(skyhook_file):
                with open(skyhook_file, 'r') as f:
                    content = f.read()
                    if "grub.conf" in content or "sysctl.conf" in content:
                        try:
                            # Parse YAML content
                            skyhook_data = yaml.safe_load(content)
                            
                            # Extract kernel tweaks if they exist
                            kernel_tweaks = {}
                            if "items" in skyhook_data and len(skyhook_data["items"]) > 0:
                                item = skyhook_data["items"][0]
                                if "spec" in item and "packages" in item["spec"] and "tuning" in item["spec"]["packages"]:
                                    config_map = item["spec"]["packages"]["tuning"].get("configMap", {})
                                    if "sysctl.conf" in config_map:
                                        kernel_tweaks["sysctl"] = config_map["sysctl.conf"]
                                    if "grub.conf" in config_map:
                                        kernel_tweaks["grub"] = config_map["grub.conf"]
                            
                            if not kernel_tweaks:
                                kernel_tweaks = "No tweaks"
                        except Exception as e:
                            print(f"Error parsing skyhook.yaml in {exp_dir}: {e}")
                            kernel_tweaks = "Error parsing tweaks"
            
            # Store the experiment data
            experiments[exp_dir] = {
                'data': df,
                'kernel_tweaks': kernel_tweaks
            }
    
    return experiments

def calculate_statistics(experiments):
    """
    Calculate summary statistics for each experiment
    """
    stats = {}
    
    metrics = [
        'step_time', 'samples_per_sec', 'cpu_percent', 'mem_percent',
        'gpu_util', 'gpu_power_W', 'gpu_temp_C', 'gpu_mem_util'
    ]
    
    for exp_name, exp_data in experiments.items():
        df = exp_data['data']
        
        # Calculate basic statistics for important metrics
        exp_stats = {}
        for metric in metrics:
            if metric in df.columns:
                exp_stats[metric] = {
                    'mean': df[metric].mean(),
                    'median': df[metric].median(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'std': df[metric].std()
                }
        
        # Calculate throughput (samples per second) over time
        if 'time_elapsed' in df.columns and 'step' in df.columns:
            total_time = df['time_elapsed'].max()
            total_steps = df['step'].max()
            exp_stats['overall_steps_per_sec'] = total_steps / total_time
        
        stats[exp_name] = exp_stats
    
    return stats

def plot_time_series(experiments, metric, title, ylabel, figsize=(12, 6), output_dir="plots"):
    """
    Plot a time series comparison of a specific metric across experiments
    """
    plt.figure(figsize=figsize)
    
    for exp_name, exp_data in experiments.items():
        df = exp_data['data']
        if metric in df.columns and 'time_elapsed' in df.columns:
            # Use a rolling mean to smooth the line
            rolling_mean = df[metric].rolling(window=10).mean()
            plt.plot(df['time_elapsed'], rolling_mean, label=exp_name)
    
    plt.title(title)
    plt.xlabel('Time Elapsed (s)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    safe_title = title.replace(' ', '_').lower()
    plt.savefig(f"{output_dir}/{safe_title}.png", dpi=300)
    plt.close()

def plot_bar_comparison(stats, metric, title, ylabel, figsize=(10, 6), output_dir="plots"):
    """
    Create a bar chart to compare a specific metric across experiments
    """
    plt.figure(figsize=figsize)
    
    exp_names = []
    values = []
    
    for exp_name, exp_stats in stats.items():
        if metric in exp_stats:
            exp_names.append(exp_name)
            values.append(exp_stats[metric]['mean'])
    
    colors = sns.color_palette('viridis', len(exp_names))
    bars = plt.bar(exp_names, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(values),
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    safe_title = title.replace(' ', '_').lower()
    plt.savefig(f"{output_dir}/{safe_title}.png", dpi=300)
    plt.close()

def plot_kernel_parameter_comparison(experiments, stats, output_dir="plots"):
    """
    Create a visualization that compares kernel parameters with performance metrics
    """
    # Extract kernel tweaks and performance metrics
    exp_names = []
    samples_per_sec = []
    step_times = []
    kernel_tweaks = []
    
    for exp_name, exp_stats in stats.items():
        if 'samples_per_sec' in exp_stats and 'step_time' in exp_stats:
            exp_names.append(exp_name)
            samples_per_sec.append(exp_stats['samples_per_sec']['mean'])
            step_times.append(exp_stats['step_time']['mean'])
            
            # Get a simplified representation of kernel tweaks
            tweaks = experiments[exp_name]['kernel_tweaks']
            if isinstance(tweaks, dict):
                tweaks_str = ", ".join([f"{k}: {len(v.splitlines())} params" for k, v in tweaks.items()])
            else:
                tweaks_str = str(tweaks)
            kernel_tweaks.append(tweaks_str)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Experiment': exp_names,
        'Samples/Sec': samples_per_sec,
        'Step Time': step_times,
        'Kernel Tweaks': kernel_tweaks
    })
    
    # Plot samples per second with kernel tweaks
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Experiment'], df['Samples/Sec'], color=sns.color_palette('viridis', len(df)))
    plt.title('Impact of Kernel Parameters on Training Throughput')
    plt.ylabel('Samples per Second (higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Add kernel tweak annotations
    for i, (bar, tweak) in enumerate(zip(bars, df['Kernel Tweaks'])):
        plt.text(bar.get_x() + bar.get_width()/2., 0.02,
                tweak, ha='center', va='bottom', rotation=90, color='black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kernel_parameter_impact.png", dpi=300)
    plt.close()

def create_summary_table(experiments, stats, output_dir="results"):
    """
    Create a summary table of all experiments and their key metrics
    """
    summary_data = []
    
    for exp_name, exp_stats in stats.items():
        row = {'Experiment': exp_name}
        
        # Add key metrics
        if 'samples_per_sec' in exp_stats:
            row['Samples/Sec'] = exp_stats['samples_per_sec']['mean']
        
        if 'step_time' in exp_stats:
            row['Step Time (s)'] = exp_stats['step_time']['mean']
        
        if 'gpu_util' in exp_stats:
            row['GPU Util (%)'] = exp_stats['gpu_util']['mean']
        
        if 'gpu_power_W' in exp_stats:
            row['GPU Power (W)'] = exp_stats['gpu_power_W']['mean']
        
        if 'cpu_percent' in exp_stats:
            row['CPU (%)'] = exp_stats['cpu_percent']['mean']
        
        if 'mem_percent' in exp_stats:
            row['Memory (%)'] = exp_stats['mem_percent']['mean']
        
        # Add kernel tweaks info
        tweaks = experiments[exp_name]['kernel_tweaks']
        if isinstance(tweaks, dict):
            row['Kernel Tweaks'] = ", ".join([f"{k}: {len(v.splitlines())} params" for k, v in tweaks.items()])
        else:
            row['Kernel Tweaks'] = str(tweaks)
        
        summary_data.append(row)
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(f'{output_dir}/experiment_summary.csv', index=False)
    
    return summary_df

def plot_performance_improvement(summary_df, output_dir="plots"):
    """
    Create a plot showing the relative performance improvement of each experiment
    compared to the baseline
    """
    # Make sure we have a baseline experiment
    if 'base' not in summary_df['Experiment'].values:
        print("Warning: No 'base' experiment found for comparison")
        return
    
    # Get baseline metrics
    baseline = summary_df[summary_df['Experiment'] == 'base']
    baseline_samples = baseline['Samples/Sec'].values[0]
    
    # Calculate improvement percentages
    improvements = []
    exp_names = []
    
    for _, row in summary_df.iterrows():
        if row['Experiment'] != 'base':
            exp_names.append(row['Experiment'])
            improvement = ((row['Samples/Sec'] - baseline_samples) / baseline_samples) * 100
            improvements.append(improvement)
    
    # Plot improvements
    plt.figure(figsize=(10, 6))
    bars = plt.bar(exp_names, improvements, color=sns.color_palette('viridis', len(exp_names)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Performance Improvement Compared to Baseline')
    plt.ylabel('Improvement in Samples/Sec (%)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_improvement.png", dpi=300)
    plt.close()

def process_experiment_set(data_dir, output_prefix):
    """
    Process a set of experiments in the given directory
    """
    # Create output directories with the prefix
    plots_dir = f"plots_{output_prefix}"
    results_dir = f"results_{output_prefix}"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load experiment data
    print(f"Loading {output_prefix} experiment data...")
    experiments = load_experiment_data(data_dir)
    
    if not experiments:
        print(f"No experiment data found in {data_dir}!")
        return
    
    print(f"Found {len(experiments)} experiments for {output_prefix}: {', '.join(experiments.keys())}")
    
    # Calculate statistics
    print(f"Calculating statistics for {output_prefix}...")
    stats = calculate_statistics(experiments)
    
    # Create summary table
    print(f"Creating summary table for {output_prefix}...")
    summary_df = create_summary_table(experiments, stats, results_dir)
    print(f"Summary for {output_prefix}:")
    print(summary_df)
    
    # Create visualizations
    print(f"Generating visualizations for {output_prefix}...")
    
    # Time series plots
    plot_time_series(experiments, 'samples_per_sec', f'{output_prefix} - Training Throughput Over Time', 'Samples per Second', output_dir=plots_dir)
    plot_time_series(experiments, 'step_time', f'{output_prefix} - Step Time Over Time', 'Step Time (s)', output_dir=plots_dir)
    plot_time_series(experiments, 'gpu_util', f'{output_prefix} - GPU Utilization Over Time', 'GPU Utilization (%)', output_dir=plots_dir)
    plot_time_series(experiments, 'gpu_power_W', f'{output_prefix} - GPU Power Consumption Over Time', 'Power (W)', output_dir=plots_dir)
    plot_time_series(experiments, 'gpu_temp_C', f'{output_prefix} - GPU Temperature Over Time', 'Temperature (°C)', output_dir=plots_dir)
    plot_time_series(experiments, 'mem_percent', f'{output_prefix} - Memory Utilization Over Time', 'Memory (%)', output_dir=plots_dir)
    plot_time_series(experiments, 'cpu_percent', f'{output_prefix} - CPU Utilization Over Time', 'CPU (%)', output_dir=plots_dir)
    
    # Bar comparison plots
    plot_bar_comparison(stats, 'samples_per_sec', f'{output_prefix} - Average Training Throughput', 'Samples per Second', output_dir=plots_dir)
    plot_bar_comparison(stats, 'step_time', f'{output_prefix} - Average Step Time', 'Step Time (s)', output_dir=plots_dir)
    
    # Kernel parameter comparison
    plot_kernel_parameter_comparison(experiments, stats, output_dir=plots_dir)
    
    # Performance improvement plot
    plot_performance_improvement(summary_df, output_dir=plots_dir)
    
    print(f"Analysis for {output_prefix} complete!")
    return summary_df

def main():
    # Define experiment sets
    experiment_sets = [
        {"data_dir": "data/one_node_one_gpu", "output_prefix": "one_node_one_gpu"},
        {"data_dir": "data/one_node_two_gpu", "output_prefix": "one_node_two_gpu"}
    ]
    
    # Process each experiment set
    all_summaries = {}
    for exp_set in experiment_sets:
        print(f"\nProcessing {exp_set['output_prefix']} experiments...\n")
        summary = process_experiment_set(exp_set["data_dir"], exp_set["output_prefix"])
        if summary is not None:
            all_summaries[exp_set["output_prefix"]] = summary
    
    # Generate cross-configuration comparison if both sets have data
    if len(all_summaries) > 1:
        print("\nGenerating cross-configuration comparison...\n")
        
        # Create a comparison of best results from each configuration
        os.makedirs("comparison", exist_ok=True)
        
        # Find best experiment in each set (highest samples/sec)
        best_configs = {}
        for config, summary in all_summaries.items():
            best_exp = summary.loc[summary['Samples/Sec'].idxmax()]
            best_configs[config] = best_exp
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(best_configs).T
        comparison_df.index.name = "Configuration"
        comparison_df.reset_index(inplace=True)
        
        # Save comparison to CSV
        comparison_df.to_csv("comparison/best_configuration_comparison.csv", index=False)
        print("Comparison of best configurations:")
        print(comparison_df)
        
        # Create comparison bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df["Configuration"], comparison_df["Samples/Sec"], 
                      color=sns.color_palette('viridis', len(comparison_df)))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(comparison_df["Samples/Sec"]),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Best Performance Comparison Across Configurations')
        plt.ylabel('Samples per Second (higher is better)')
        plt.tight_layout()
        plt.savefig("comparison/best_configuration_comparison.png", dpi=300)
        plt.close()
    
    print("\nAll analyses complete! Results are in the respective directories.")

if __name__ == "__main__":
    main()
