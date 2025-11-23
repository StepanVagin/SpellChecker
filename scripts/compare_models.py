#!/usr/bin/env python3
"""
Compare evaluation results between baseline and updated models.

Usage:
    python scripts/compare_models.py \
        --baseline evaluation_results_baseline.json \
        --updated evaluation_results.json
"""

import argparse
import json
import os


def compare_models(
    baseline_results: str,
    updated_results: str
):
    """Compare baseline and updated model performance."""
    
    if not os.path.exists(baseline_results):
        print(f"Error: Baseline results file not found: {baseline_results}")
        return
    
    if not os.path.exists(updated_results):
        print(f"Error: Updated results file not found: {updated_results}")
        return
    
    with open(baseline_results, 'r') as f:
        baseline = json.load(f)
    
    with open(updated_results, 'r') as f:
        updated = json.load(f)
    
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Baseline: {baseline_results}")
    print(f"Updated:  {updated_results}")
    
    datasets_to_compare = set(baseline.keys()) & set(updated.keys())
    
    if not datasets_to_compare:
        print("\nNo common datasets found to compare!")
        return
    
    for dataset in sorted(datasets_to_compare):
        print(f"\n{dataset.upper().replace('_', ' ')}:")
        print(f"{'-'*70}")
        print(f"{'Metric':<20} {'Baseline':<15} {'Updated':<15} {'Change':<15} {'% Change':<15}")
        print(f"{'-'*70}")
        
        baseline_data = baseline[dataset]
        updated_data = updated[dataset]
        
        metrics_to_compare = ['exact_match', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_compare:
            if metric not in baseline_data or metric not in updated_data:
                continue
            
            base_val = baseline_data[metric]
            upd_val = updated_data[metric]
            change = upd_val - base_val
            
            if base_val != 0:
                pct_change = (change / base_val) * 100
            else:
                pct_change = 0.0 if change == 0 else float('inf')
            
            change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
            pct_change_str = f"{pct_change:+.1f}%" if abs(pct_change) < 1000 else "N/A"
            
            # Color coding: green for improvement, red for degradation
            if change > 0:
                indicator = "↑"
            elif change < 0:
                indicator = "↓"
            else:
                indicator = "="
            
            print(f"{metric:<20} {base_val:>13.2f}% {upd_val:>13.2f}% "
                  f"{change_str:>13} {pct_change_str:>13} {indicator}")
        
        # Show detailed statistics if available
        if 'true_positives' in baseline_data and 'true_positives' in updated_data:
            print(f"\nDetailed Statistics:")
            print(f"{'-'*70}")
            
            stats = ['true_positives', 'false_positives', 'false_negatives', 
                     'exact_matches', 'total_sentences']
            
            for stat in stats:
                if stat in baseline_data and stat in updated_data:
                    base_stat = baseline_data[stat]
                    upd_stat = updated_data[stat]
                    change_stat = upd_stat - base_stat
                    change_str = f"{change_stat:+,d}" if change_stat != 0 else "0"
                    
                    print(f"  {stat:<20} {base_stat:>10,} → {upd_stat:>10,} ({change_str})")
    
    print(f"\n{'='*70}")
    print("Summary:")
    
    # Calculate overall improvement
    total_improvements = 0
    total_degradations = 0
    
    for dataset in datasets_to_compare:
        baseline_data = baseline[dataset]
        updated_data = updated[dataset]
        
        for metric in ['exact_match', 'precision', 'recall', 'f1_score']:
            if metric in baseline_data and metric in updated_data:
                change = updated_data[metric] - baseline_data[metric]
                if change > 0:
                    total_improvements += 1
                elif change < 0:
                    total_degradations += 1
    
    print(f"  Metrics improved:   {total_improvements}")
    print(f"  Metrics degraded:   {total_degradations}")
    
    if total_improvements > total_degradations:
        print(f"\n✓ Overall: Model update shows improvement!")
    elif total_degradations > total_improvements:
        print(f"\n✗ Overall: Model update shows degradation.")
    else:
        print(f"\n= Overall: Model update shows mixed results.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between model versions"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="evaluation_results_baseline.json",
        help="Baseline evaluation results JSON file"
    )
    
    parser.add_argument(
        "--updated",
        type=str,
        default="evaluation_results.json",
        help="Updated evaluation results JSON file"
    )
    
    args = parser.parse_args()
    
    compare_models(
        baseline_results=args.baseline,
        updated_results=args.updated
    )


if __name__ == "__main__":
    main()

