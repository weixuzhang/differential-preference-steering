#!/usr/bin/env python3
"""
Quick test script for preference head detection.
Uses a small sample size for rapid testing.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preference_head_detection import PreferenceHeadConfig, PreferenceHeadDetector


def test_detection(model_path: str, num_samples: int = 10):
    """
    Test preference head detection with a small number of samples.
    
    Args:
        model_path: Path to model
        num_samples: Number of samples to use for testing
    """
    print("="*80)
    print("PREFERENCE HEAD DETECTION - TEST MODE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test samples: {num_samples}")
    print("="*80)
    print()
    
    # Create config for testing
    config = PreferenceHeadConfig(
        model_path=model_path,
        task="LaMP-1",
        num_samples=num_samples,
        device="cuda",
        torch_dtype="bfloat16",
        top_percent=0.1,  # Use 10% for testing (more lenient)
        save_dir="/scratch/weixuz/preference_head/test_preference_scores"
    )
    
    # Initialize detector
    print("Initializing detector...")
    detector = PreferenceHeadDetector(config)
    
    # Run detection
    print("\nRunning detection (this may take a few minutes)...")
    detector.detect_preference_heads()
    
    # Save results
    print("\nSaving results...")
    detector.save_results()
    
    # Get top heads
    top_heads = detector.get_top_preference_heads()
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"Detected {len(top_heads)} preference heads")
    print("\nTop 5 preference heads:")
    ranked = detector.rank_heads()
    for i, ((layer, head), avg_pcs) in enumerate(ranked[:5], 1):
        print(f"  {i}. Layer {layer}, Head {head}: PCS = {avg_pcs:.6f}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Run full detection: python preference_head_detection.py --model_path <path> --num_samples 400")
    print("2. Validate heads: python validate_preference_heads.py --model_path <path> --preference_heads_file <file>")
    print("3. Use in DPS: Load preference heads from saved JSON file")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test preference head detection")
    parser.add_argument("--model_path", type=str, 
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Path to model")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    test_detection(args.model_path, args.num_samples)

