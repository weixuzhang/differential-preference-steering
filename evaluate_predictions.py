#!/usr/bin/env python3
"""
Comprehensive evaluation script for LAMP predictions.
Supports all LAMP tasks (1-7) with proper metrics for each task type.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
banditpr_root = os.environ.get("BANDITPR_ROOT")
if banditpr_root:
    sys.path.append(str(Path(banditpr_root) / "src"))

import evaluate

try:
    import datasets.config as _datasets_config
    if not hasattr(_datasets_config, "importlib_metadata"):
        import importlib.metadata as importlib_metadata
        _datasets_config.importlib_metadata = importlib_metadata
except Exception:
    pass


def get_labels(task: str) -> list[str]:
    if task == "LaMP-1":
        return ["[1]", "[2]"]
    if task == "LaMP-2":
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    if task == "LaMP-3":
        return ["1", "2", "3", "4", "5"]
    raise ValueError(f"Not a classification or regression task: {task}")


def create_metric(task: str, average: bool = True):
    if task in {"LaMP-1", "LaMP-2"}:
        return _create_classification_metric(get_labels(task), average)
    if task in {"LaMP-3"}:
        return _create_regression_metric(average)
    if task in {"LaMP-4", "LaMP-5", "LaMP-6", "LaMP-7"}:
        return _create_generation_metric(average)
    if task in {"LongLaMP-1", "LongLaMP-2", "LongLaMP-3", "LongLaMP-4"}:
        return _create_generation_metric(average)
    raise ValueError(f"Invalid task: {task}")


def _create_classification_metric(labels: list[str], average: bool):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def map_to_label_index(string: str) -> int:
        try:
            return labels.index(string.strip())
        except ValueError:
            return -1

    def classification_metric(predictions: list[str], targets: list[str]):
        predictions = [map_to_label_index(prediction) for prediction in predictions]
        targets = [map_to_label_index(target) for target in targets]

        if average:
            accuracy_results = accuracy_metric.compute(
                predictions=predictions, references=targets
            )
            f1_results = f1_metric.compute(
                predictions=predictions,
                references=targets,
                labels=list(range(len(labels))),
                average="macro",
            )
            return {"accuracy": accuracy_results["accuracy"], "f1": f1_results["f1"]}

        accuracy = [
            int(prediction == target)
            for prediction, target in zip(predictions, targets)
        ]
        return {"accuracy": accuracy}

    return classification_metric


def _create_regression_metric(average: bool):
    mae_metric = evaluate.load("mae")
    mse_metric = evaluate.load("mse")

    def map_to_float(prediction: str, target: str) -> float:
        try:
            return float(prediction)
        except ValueError:
            target = float(target)
            return 1.0 if abs(1 - target) > abs(5 - target) else 5.0

    def regression_metric(predictions: list[str], targets: list[str]):
        predictions = [
            map_to_float(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
        targets = [float(target) for target in targets]

        if average:
            mae_results = mae_metric.compute(
                predictions=predictions, references=targets
            )
            rmse_results = mse_metric.compute(
                predictions=predictions, references=targets, squared=False
            )
            return {"mae": mae_results["mae"], "rmse": rmse_results["mse"]}

        return {
            "mae": [
                abs(prediction - target)
                for prediction, target in zip(predictions, targets)
            ]
        }

    return regression_metric


def _create_generation_metric(average: bool):
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    def generation_metric(predictions: list[str], targets: list[str]):
        predictions = [prediction.strip() for prediction in predictions]
        targets = [[target.strip()] for target in targets]

        rouge_results = rouge_metric.compute(
            predictions=predictions,
            references=targets,
            rouge_types=["rouge1", "rougeL"],
            use_aggregator=average,
        )

        if average:
            meteor_results = meteor_metric.compute(
                predictions=predictions, references=targets
            )
        else:
            meteor_results = {
                "meteor": [
                    meteor_metric.compute(
                        predictions=[prediction], references=[target]
                    )["meteor"]
                    for prediction, target in zip(predictions, targets)
                ]
            }

        return {
            "rouge-1": rouge_results["rouge1"],
            "rouge-L": rouge_results["rougeL"],
            "meteor": meteor_results["meteor"],
        }

    return generation_metric


def load_predictions(file_path: str) -> List[Dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except:
                continue
    return predictions


def extract_task_name(predictions: List[Dict]) -> str:
    """Extract task name from predictions."""
    if not predictions:
        return "Unknown"
    
    task_field = predictions[0].get('task', 'Unknown')
    if isinstance(task_field, list) and len(task_field) > 0:
        return task_field[0]
    return str(task_field)


def evaluate_predictions(file_path: str) -> Dict[str, float]:
    """
    Evaluate predictions using the appropriate metric for the task.
    
    Returns:
        Dict with metrics appropriate for the task type
    """
    predictions = load_predictions(file_path)
    if not predictions:
        return {"error": "No predictions found"}
    
    # Extract task name
    task = extract_task_name(predictions)
    
    # Extract predictions and ground truth
    predicted_answers = []
    ground_truth = []
    
    for sample in predictions:
        # Get predicted answer (string)
        pred = sample.get('predicted_answer', '')
        predicted_answers.append(pred)
        
        # Get ground truth - handle nested list structure [[answer]]
        gt = sample.get('answers', '')
        if isinstance(gt, list):
            while isinstance(gt, list) and len(gt) > 0:
                gt = gt[0]
        ground_truth.append(str(gt))
    
    # Compute metrics using BanditPR's metric function
    try:
        lamp_metric = create_metric(task, average=True)
        results = lamp_metric(predicted_answers, ground_truth)
        results['task'] = task
        results['total_samples'] = len(predictions)
        return results
    except Exception as e:
        print(f"Warning: Error computing metrics for {task}: {e}")
        # Fallback to basic accuracy
        correct = sum(1 for p, g in zip(predicted_answers, ground_truth) if p.strip() == g.strip())
        return {
            'task': task,
            'total_samples': len(predictions),
            'accuracy': correct / len(predictions) if predictions else 0.0,
            'correct': correct,
            'error': str(e)
        }


def find_latest_predictions(outputs_dir: str = "outputs") -> Dict[str, str]:
    """
    Find the most recent prediction files for each task and method combination.
    
    Returns:
        Dict mapping (task, method) -> file_path
    """
    outputs_path = Path(outputs_dir)
    pred_files = {}
    
    # Find all prediction files, excluding _eval.json files
    for pred_file in outputs_path.rglob("pred_LAMP_*.json"):
        # Skip evaluation summary files
        if pred_file.name.endswith('_eval.json'):
            continue
            
        # Extract task and method from filename
        filename = pred_file.name
        # Format: pred_LAMP_X_ModelName__Method.json
        parts = filename.replace('pred_', '').replace('.json', '').split('__')
        if len(parts) == 2:
            task_model = parts[0]  # e.g., "LAMP_1_LLaMA3-8b-Instruct"
            method = parts[1]      # e.g., "DeCoReEntropy" or "Baseline"
            
            # Extract task number
            task_parts = task_model.split('_')
            if len(task_parts) >= 2 and task_parts[0] == 'LAMP':
                task_num = task_parts[1]
                key = (f"LAMP-{task_num}", method)
                
                # Keep the most recent file (based on modification time)
                if key not in pred_files or pred_file.stat().st_mtime > Path(pred_files[key]).stat().st_mtime:
                    pred_files[key] = str(pred_file)
    
    return pred_files


def find_newest_prediction(outputs_dir: str = "outputs") -> str | None:
    """Find the most recent prediction file (by mtime)."""
    outputs_path = Path(outputs_dir)
    pred_files = [
        pred_file
        for pred_file in outputs_path.rglob("pred_*.json")
        if not pred_file.name.endswith("_eval.json")
    ]
    if not pred_files:
        return None
    newest_file = max(pred_files, key=lambda p: p.stat().st_mtime)
    return str(newest_file)


def print_results_table(all_results: Dict[Tuple[str, str], Dict]):
    """Print results in a nice formatted table."""
    print("\n" + "="*80)
    print("LAMP Benchmark Evaluation Results")
    print("="*80)
    
    # Group by task
    tasks = sorted(set(task for task, _ in all_results.keys()))
    methods = sorted(set(method for _, method in all_results.keys()))
    
    for task in tasks:
        print(f"\n{'─'*80}")
        print(f"📊 {task}")
        print(f"{'─'*80}")
        
        for method in methods:
            key = (task, method)
            if key not in all_results:
                continue
            
            results = all_results[key]
            print(f"\n  {method}:")
            
            # Print metrics based on task type
            if 'accuracy' in results:
                print(f"    Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            if 'f1' in results:
                print(f"    F1 Score: {results['f1']:.4f}")
            if 'mae' in results:
                print(f"    MAE:      {results['mae']:.4f}")
            if 'rmse' in results:
                print(f"    RMSE:     {results['rmse']:.4f}")
            if 'rouge-1' in results:
                print(f"    ROUGE-1:  {results['rouge-1']:.4f}")
            if 'rouge-L' in results:
                print(f"    ROUGE-L:  {results['rouge-L']:.4f}")
            if 'meteor' in results:
                print(f"    METEOR:   {results['meteor']:.4f}")
            
            print(f"    Samples:  {results.get('total_samples', 'N/A')}")
            
            if 'error' in results:
                print(f"    ⚠️  Warning: {results['error']}")
    
    print(f"\n{'='*80}\n")


def main(outputs_dir: str = "outputs", specific_file: str = None):
    """
    Main evaluation function.
    
    Args:
        outputs_dir: Directory containing output predictions
        specific_file: Optional path to a specific prediction file to evaluate
    """
    if specific_file:
        # Evaluate a specific file
        if not os.path.exists(specific_file):
            print(f"Error: File not found: {specific_file}")
            return
        
        print(f"Evaluating: {specific_file}")
        results = evaluate_predictions(specific_file)
        print(f"\nResults:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        # Find and evaluate all latest predictions
        pred_files = find_latest_predictions(outputs_dir)
        
        if not pred_files:
            print(f"No prediction files found in {outputs_dir}")
            return
        
        print(f"Found {len(pred_files)} prediction files to evaluate...")
        
        all_results = {}
        for (task, method), file_path in sorted(pred_files.items()):
            print(f"  Evaluating {task} - {method}... ", end='', flush=True)
            results = evaluate_predictions(file_path)
            all_results[(task, method)] = results
            print("✓")
        
        # Print comprehensive results
        print_results_table(all_results)
        
        # Save results to JSON
        output_file = Path(outputs_dir) / "evaluation_summary.json"
        with open(output_file, 'w') as f:
            # Convert tuple keys to strings for JSON
            json_results = {f"{task}_{method}": results for (task, method), results in all_results.items()}
            json.dump(json_results, f, indent=2)
        print(f"📁 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LAMP predictions")
    parser.add_argument('--outputs-dir', default='outputs', help='Directory containing predictions')
    parser.add_argument('--file', help='Evaluate a specific prediction file')
    parser.add_argument('--latest', action='store_true', help='Evaluate only the most recent prediction file')
    
    args = parser.parse_args()
    if args.latest and not args.file:
        latest_path = find_newest_prediction(args.outputs_dir)
        if not latest_path:
            print(f"No prediction files found in {args.outputs_dir}")
            sys.exit(0)
        main(args.outputs_dir, latest_path)
    else:
        main(args.outputs_dir, args.file)
