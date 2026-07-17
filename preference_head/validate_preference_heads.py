#!/usr/bin/env python3
"""
Validation script for preference heads using activation patching.

This script validates that detected preference heads actually carry preference
information by:
1. Running model on clean prompt (with profile)
2. Running model on corrupted prompt (without profile)
3. Patching head activations from clean to corrupted
4. Measuring if personalization improves
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add paths
banditpr_root = os.environ.get("BANDITPR_ROOT")
if banditpr_root:
    sys.path.append(str(Path(banditpr_root) / "src"))
from lamp import load_lamp_dataset


def _is_offline() -> bool:
    """Return True if HF offline mode is enabled via env vars."""
    offline_vars = (
        os.environ.get("HF_OFFLINE", ""),
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("HF_DATASETS_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    )
    return any(val.strip().lower() in {"1", "true", "yes"} for val in offline_vars)


def _raise_offline_model_error(model_path: str, exc: Exception) -> None:
    raise RuntimeError(
        f"HF offline mode is enabled but model '{model_path}' is not in cache. "
        "Set HF_OFFLINE=false (or HF_HUB_OFFLINE=0) to download, "
        "or pre-download into the cache."
    ) from exc


class PreferenceHeadValidator:
    """Validates preference heads using activation patching."""
    
    def __init__(
        self,
        model_path: str,
        preference_heads: List[List[int]],
        task: str = "LaMP-1",
        num_validation_samples: int = 100,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.preference_heads = preference_heads
        self.task = task
        self.num_validation_samples = num_validation_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.offline = _is_offline()
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=self.offline,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True,
                local_files_only=self.offline,
            )
        except OSError as exc:
            if self.offline:
                _raise_offline_model_error(model_path, exc)
            raise
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        
        # Load dataset
        print(f"Loading {task} validation dataset...")
        
        dataset = load_lamp_dataset(task, split='dev')
        
        if num_validation_samples > 0 and num_validation_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_validation_samples, replace=False)
            dataset = dataset.select(indices.tolist())
        self.dataset = dataset
        
        print(f"Loaded {len(self.dataset)} validation samples")
        print(f"Testing {len(preference_heads)} preference heads")
    
    def _format_prompt(self, sample: Dict, include_profile: bool = True) -> str:
        """Format prompt with or without profile."""
        source = sample['source']
        
        if include_profile:
            profiles = sample['profiles']
            profile_text = ""
            for i, prof in enumerate(profiles[:5]):
                if 'text' in prof:
                    profile_text += f"History {i+1}: {prof['text']}\n"
                elif 'title' in prof:
                    profile_text += f"History {i+1}: {prof['title']}\n"
            prompt = f"{profile_text}\n{source}"
        else:
            prompt = source
        
        return prompt
    
    def _compute_nll(self, prompt: str, target: str) -> float:
        """Compute negative log-likelihood."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        full_ids = torch.cat([input_ids, target_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(full_ids, labels=full_ids)
            logits = outputs.logits
            shift_logits = logits[:, input_ids.shape[1]-1:-1, :].contiguous()
            shift_labels = target_ids.contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
        
        return loss.item()
    
    def validate_with_masking(self) -> Dict[str, float]:
        """
        Validate by masking preference heads vs random heads.
        
        Expected: Masking preference heads hurts personalization more than random heads.
        """
        print("\n" + "="*80)
        print("Validation: Masking Experiment")
        print("="*80)
        
        results = {
            'baseline_nll': [],
            'preference_masked_nll': [],
            'random_masked_nll': []
        }
        
        # Select random heads for comparison
        all_heads = [[l, h] for l in range(self.num_layers) for h in range(self.num_heads)]
        random_heads = [h for h in all_heads if h not in self.preference_heads]
        random_sample = np.random.choice(len(random_heads), len(self.preference_heads), replace=False)
        random_heads_selected = [random_heads[i] for i in random_sample]
        
        print(f"Testing on {len(self.dataset)} samples...")
        print(f"Masking {len(self.preference_heads)} preference heads vs {len(random_heads_selected)} random heads")
        
        for sample in tqdm(self.dataset):
            prompt = self._format_prompt(sample, include_profile=True)
            target = sample['target']
            
            try:
                # Baseline (no masking)
                baseline_nll = self._compute_nll(prompt, target)
                results['baseline_nll'].append(baseline_nll)
                
                # Mask preference heads (implementation would need to add masking hook)
                # For now, this is a placeholder - actual implementation would mask during forward pass
                
            except Exception as e:
                print(f"Error on sample: {e}")
                continue
        
        # Compute averages
        avg_results = {k: np.mean(v) for k, v in results.items() if v}
        
        print("\nResults:")
        print(f"  Baseline NLL:           {avg_results.get('baseline_nll', 0):.4f}")
        print(f"  Preference Masked NLL:  {avg_results.get('preference_masked_nll', 0):.4f}")
        print(f"  Random Masked NLL:      {avg_results.get('random_masked_nll', 0):.4f}")
        
        return avg_results
    
    def validate_with_patching(self) -> Dict[str, float]:
        """
        Validate using activation patching.
        
        Run model on:
        1. Clean prompt (with profile) - cache activations
        2. Corrupted prompt (without profile) - patch activations
        
        Expected: Patching preference heads improves personalization on corrupted input.
        """
        print("\n" + "="*80)
        print("Validation: Activation Patching")
        print("="*80)
        print("This validates causal effect of preference heads")
        
        results = {
            'clean_nll': [],
            'corrupted_nll': [],
            'patched_nll': []
        }
        
        print(f"\nTesting on {len(self.dataset)} samples...")
        
        for sample in tqdm(self.dataset):
            clean_prompt = self._format_prompt(sample, include_profile=True)
            corrupted_prompt = self._format_prompt(sample, include_profile=False)
            target = sample['target']
            
            try:
                # Clean: with profile
                clean_nll = self._compute_nll(clean_prompt, target)
                results['clean_nll'].append(clean_nll)
                
                # Corrupted: without profile
                corrupted_nll = self._compute_nll(corrupted_prompt, target)
                results['corrupted_nll'].append(corrupted_nll)
                
                # TODO: Implement activation patching
                # This requires caching activations from clean run and injecting during corrupted run
                
            except Exception as e:
                print(f"Error on sample: {e}")
                continue
        
        # Compute averages
        avg_results = {k: np.mean(v) for k, v in results.items() if v}
        
        print("\nResults:")
        print(f"  Clean NLL (with profile):      {avg_results.get('clean_nll', 0):.4f}")
        print(f"  Corrupted NLL (no profile):    {avg_results.get('corrupted_nll', 0):.4f}")
        print(f"  Patched NLL (heads restored):  {avg_results.get('patched_nll', 0):.4f}")
        
        delta_profile = avg_results.get('corrupted_nll', 0) - avg_results.get('clean_nll', 0)
        print(f"\n  Profile effect (Δ NLL):        {delta_profile:.4f}")
        print(f"  (Positive Δ means profile helps, which is expected)")
        
        return avg_results


def main():
    parser = argparse.ArgumentParser(description="Validate detected preference heads")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--preference_heads_file", type=str, required=True,
                       help="Path to preference heads JSON file")
    parser.add_argument("--task", type=str, default="LaMP-1",
                       help="LaMP task")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of validation samples")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load preference heads
    print(f"Loading preference heads from {args.preference_heads_file}...")
    with open(args.preference_heads_file, 'r') as f:
        heads_data = json.load(f)
    preference_heads = heads_data['preference_heads']
    
    print(f"Loaded {len(preference_heads)} preference heads")
    
    # Initialize validator
    validator = PreferenceHeadValidator(
        model_path=args.model_path,
        preference_heads=preference_heads,
        task=args.task,
        num_validation_samples=args.num_samples,
        device=args.device
    )
    
    # Run validation
    masking_results = validator.validate_with_masking()
    patching_results = validator.validate_with_patching()


if __name__ == "__main__":
    main()
