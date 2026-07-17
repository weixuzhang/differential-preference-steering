#!/usr/bin/env python3
"""
Ablation Study Analysis Script
Analyzes the effect of different numbers of preference heads on DPS performance
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re

def load_prediction_file(filepath: Path) -> Dict:
    """Load a prediction JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_num_heads_from_filename(filename: str) -> int:
    """Extract number of heads from filename like pred_LAMP_1_LLaMA3-8b-Instruct__DPS_heads40.json"""
    match = re.search(r'heads(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def extract_task_from_filename(filename: str) -> str:
    """Extract task name from filename."""
    # Match patterns like pred_LAMP_1_ or pred_LongLaMP_1_
    match = re.search(r'pred_((?:Long)?LaMP_\d+)_', filename)
    if match:
        task = match.group(1).replace('_', '-')
        return task
    return None

def get_primary_metric(task: str, data: Dict) -> Tuple[str, float]:
    """Get the primary metric for a task."""
    task_num = task.split('-')[-1]
    
    # Classification tasks
    if task in ['LaMP-1', 'LaMP-2']:
        return 'accuracy', data.get('accuracy', 0.0)
    
    # Regression task
    elif task == 'LaMP-3':
        return 'mae', data.get('mae', float('inf'))
    
    # Generation tasks
    else:
        return 'rouge-L', data.get('rouge-L', 0.0)

def analyze_ablation_results(outputs_dir: Path) -> Dict:
    """Analyze ablation study results."""
    
    # Find all ablation prediction files
    ablation_files = list(outputs_dir.glob("pred_*__DPS*.json"))
    ablation_files = [f for f in ablation_files if 'heads' in f.name or '_eval' not in f.name]
    
    print(f"Found {len(ablation_files)} ablation prediction files")
    
    # Organize results by task and number of heads
    results = defaultdict(lambda: defaultdict(dict))
    
    for filepath in ablation_files:
        data = load_prediction_file(filepath)
        if data is None:
            continue
        
        task = extract_task_from_filename(filepath.name)
        num_heads = extract_num_heads_from_filename(filepath.name)
        
        if task is None or num_heads is None:
            continue
        
        # Get primary metric
        metric_name, metric_value = get_primary_metric(task, data)
        
        results[task][num_heads] = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'all_metrics': data
        }
        
        print(f"  {task} with {num_heads} heads: {metric_name}={metric_value:.4f}")
    
    return dict(results)

def find_optimal_heads(results: Dict) -> Dict:
    """Find optimal number of heads for each task."""
    optimal = {}
    
    for task, head_results in results.items():
        if not head_results:
            continue
        
        # Get metric name (should be same for all)
        metric_name = list(head_results.values())[0]['metric_name']
        
        # For MAE, lower is better; for others, higher is better
        if metric_name == 'mae':
            best_heads = min(head_results.items(), key=lambda x: x[1]['metric_value'])
        else:
            best_heads = max(head_results.items(), key=lambda x: x[1]['metric_value'])
        
        num_heads, best_result = best_heads
        
        optimal[task] = {
            'optimal_num_heads': num_heads,
            'metric_name': metric_name,
            'metric_value': best_result['metric_value']
        }
    
    return optimal

def generate_summary(results: Dict, optimal: Dict) -> Dict:
    """Generate summary statistics."""
    summary = {
        'total_tasks': len(results),
        'head_counts_tested': sorted(set(
            num_heads 
            for task_results in results.values() 
            for num_heads in task_results.keys()
        )),
        'optimal_per_task': optimal,
        'detailed_results': {}
    }
    
    # Detailed results by task
    for task, head_results in results.items():
        task_summary = {
            'metric_name': list(head_results.values())[0]['metric_name'] if head_results else None,
            'results_by_num_heads': {}
        }
        
        for num_heads in sorted(head_results.keys()):
            task_summary['results_by_num_heads'][num_heads] = {
                'metric_value': head_results[num_heads]['metric_value']
            }
        
        summary['detailed_results'][task] = task_summary
    
    return summary

def print_summary_table(results: Dict, optimal: Dict):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY: Number of Preference Heads")
    print("="*80)
    print()
    
    # Group by task type
    classification_tasks = ['LaMP-1', 'LaMP-2']
    regression_tasks = ['LaMP-3']
    short_gen_tasks = ['LaMP-4', 'LaMP-5', 'LaMP-7']
    long_gen_tasks = ['LongLaMP-1', 'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4']
    
    for task_group, task_list in [
        ("Classification Tasks", classification_tasks),
        ("Regression Task", regression_tasks),
        ("Short Generation Tasks", short_gen_tasks),
        ("Long Generation Tasks", long_gen_tasks)
    ]:
        print(f"\n{task_group}:")
        print("-" * 80)
        
        for task in task_list:
            if task not in results:
                print(f"  {task:15s} - No results")
                continue
            
            opt = optimal.get(task, {})
            metric_name = opt.get('metric_name', 'N/A')
            metric_value = opt.get('metric_value', 0.0)
            num_heads = opt.get('optimal_num_heads', 'N/A')
            
            print(f"  {task:15s} - Optimal: {num_heads:3d} heads ({metric_name}={metric_value:.4f})")
            
            # Show performance across different head counts
            head_results = results[task]
            perf_str = "    Performance: "
            for nh in sorted(head_results.keys()):
                val = head_results[nh]['metric_value']
                marker = "⭐" if nh == num_heads else " "
                perf_str += f"{nh}h:{val:.3f}{marker} "
            print(perf_str)
    
    print("\n" + "="*80)
    print()

def main():
    """Main analysis function."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("Error: outputs/ directory not found")
        sys.exit(1)
    
    print("Analyzing ablation study results...")
    print()
    
    # Analyze results
    results = analyze_ablation_results(outputs_dir)
    
    if not results:
        print("No ablation results found!")
        sys.exit(1)
    
    # Find optimal heads
    optimal = find_optimal_heads(results)
    
    # Generate summary
    summary = generate_summary(results, optimal)
    
    # Print summary table
    print_summary_table(results, optimal)
    
    # Save to file
    output_file = outputs_dir / "ablation_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Ablation analysis saved to: {output_file}")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    optimal_counts = [opt['optimal_num_heads'] for opt in optimal.values()]
    if optimal_counts:
        avg_optimal = sum(optimal_counts) / len(optimal_counts)
        print(f"  Average optimal number of heads: {avg_optimal:.1f}")
        print(f"  Range: {min(optimal_counts)} - {max(optimal_counts)} heads")
        print()
        print(f"  Recommendation: Use {int(round(avg_optimal))} preference heads for best overall performance")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

