#!/usr/bin/env python3
"""
Visualization script for preference heads.
Creates heatmaps and comparison plots.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_preference_scores(pcs_file: str) -> Dict[Tuple[int, int], List[float]]:
    """Load PCS scores from JSON file."""
    with open(pcs_file, 'r') as f:
        scores_dict = json.load(f)
    
    # Convert string keys back to tuples
    scores = {}
    for key, values in scores_dict.items():
        layer, head = map(int, key.split('-'))
        scores[(layer, head)] = values
    
    return scores


def create_pcs_heatmap(
    scores: Dict[Tuple[int, int], List[float]],
    output_file: str,
    title: str = "Preference Contribution Score (PCS) Heatmap"
):
    """Create heatmap of average PCS scores."""
    
    # Get dimensions
    max_layer = max(layer for layer, _ in scores.keys())
    max_head = max(head for _, head in scores.keys())
    
    # Create matrix
    pcs_matrix = np.zeros((max_layer + 1, max_head + 1))
    for (layer, head), values in scores.items():
        pcs_matrix[layer, head] = np.mean(values)
    
    # Create heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        pcs_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'Average PCS'},
        xticklabels=range(max_head + 1),
        yticklabels=range(max_layer + 1)
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Head Index', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Heatmap saved to: {output_file}")
    plt.close()


def create_top_heads_bar_plot(
    ranked_file: str,
    output_file: str,
    top_k: int = 20,
    title: str = "Top Preference Heads by PCS"
):
    """Create bar plot of top preference heads."""
    
    # Load ranked heads
    with open(ranked_file, 'r') as f:
        data = json.load(f)
    
    ranked_heads = data['ranked_heads'][:top_k]
    
    # Extract data
    labels = [f"L{h['layer']}-H{h['head']}" for h in ranked_heads]
    pcs_values = [h['avg_pcs'] for h in ranked_heads]
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(labels)), pcs_values, color='steelblue')
    
    # Color top 5 differently
    for i in range(min(5, len(bars))):
        bars[i].set_color('darkred')
    
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Average PCS', fontsize=12)
    plt.ylabel('Head (Layer-Head)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Bar plot saved to: {output_file}")
    plt.close()


def create_layer_distribution(
    ranked_file: str,
    output_file: str,
    title: str = "Preference Head Distribution Across Layers"
):
    """Create histogram showing distribution of preference heads across layers."""
    
    # Load ranked heads
    with open(ranked_file, 'r') as f:
        data = json.load(f)
    
    # Get top 10% of heads
    num_top = max(10, int(len(data['ranked_heads']) * 0.1))
    top_heads = data['ranked_heads'][:num_top]
    
    # Extract layers
    layers = [h['layer'] for h in top_heads]
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(layers, bins=range(max(layers) + 2), alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Number of Preference Heads', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Layer distribution saved to: {output_file}")
    plt.close()


def compare_with_retrieval_heads(
    preference_file: str,
    retrieval_file: str,
    output_file: str
):
    """
    Compare preference heads vs retrieval heads.
    
    Shows overlap and differences between the two head types.
    """
    # Load preference heads
    with open(preference_file, 'r') as f:
        pref_data = json.load(f)
    preference_heads = set(tuple(h) for h in pref_data['preference_heads'])
    
    # Load retrieval heads
    with open(retrieval_file, 'r') as f:
        retrieval_data = json.load(f)
    
    # Parse retrieval heads (format: {"layer-head": [scores]})
    retrieval_heads = set()
    for key, scores in retrieval_data.items():
        layer, head = map(int, key.split('-'))
        # Take top scoring retrieval heads
        if np.mean(scores) > 0.3:  # Threshold for retrieval heads
            retrieval_heads.add((layer, head))
    
    # Compute overlap
    overlap = preference_heads & retrieval_heads
    only_preference = preference_heads - retrieval_heads
    only_retrieval = retrieval_heads - preference_heads
    
    # Create Venn diagram-style plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Bar plot
    categories = ['Preference\nOnly', 'Both', 'Retrieval\nOnly']
    counts = [len(only_preference), len(overlap), len(only_retrieval)]
    colors = ['steelblue', 'purple', 'coral']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Heads', fontsize=12)
    ax.set_title('Preference Heads vs Retrieval Heads\nOverlap Analysis', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {output_file}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Preference vs Retrieval Heads - Summary")
    print("="*60)
    print(f"Preference heads only:  {len(only_preference)}")
    print(f"Overlap (both types):   {len(overlap)}")
    print(f"Retrieval heads only:   {len(only_retrieval)}")
    print(f"Total preference:       {len(preference_heads)}")
    print(f"Total retrieval:        {len(retrieval_heads)}")
    overlap_pct = 100 * len(overlap) / len(preference_heads) if preference_heads else 0
    print(f"Overlap percentage:     {overlap_pct:.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize preference head detection results")
    
    parser.add_argument("--pcs_file", type=str,
                       help="Path to PCS scores JSON file")
    parser.add_argument("--ranked_file", type=str,
                       help="Path to ranked heads JSON file")
    parser.add_argument("--retrieval_file", type=str,
                       help="Optional: Path to retrieval heads file for comparison")
    parser.add_argument("--output_dir", type=str, default="results/preference_head/visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("Preference Head Visualization")
    print("="*80)
    
    # Create heatmap
    if args.pcs_file:
        print("\nCreating PCS heatmap...")
        scores = load_preference_scores(args.pcs_file)
        create_pcs_heatmap(
            scores,
            output_dir / "pcs_heatmap.png"
        )
    
    # Create bar plot
    if args.ranked_file:
        print("\nCreating top heads bar plot...")
        create_top_heads_bar_plot(
            args.ranked_file,
            output_dir / "top_heads_bar.png"
        )
        
        print("\nCreating layer distribution...")
        create_layer_distribution(
            args.ranked_file,
            output_dir / "layer_distribution.png"
        )
    
    # Compare with retrieval heads
    if args.retrieval_file and args.ranked_file:
        print("\nComparing with retrieval heads...")
        # Need to load preference heads file
        base_name = Path(args.ranked_file).stem.replace('_ranked', '_top_heads')
        pref_heads_file = Path(args.ranked_file).parent / f"{base_name}.json"
        
        if pref_heads_file.exists():
            compare_with_retrieval_heads(
                str(pref_heads_file),
                args.retrieval_file,
                output_dir / "preference_vs_retrieval.png"
            )
        else:
            print(f"Warning: Could not find preference heads file: {pref_heads_file}")
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print(f"📁 Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

