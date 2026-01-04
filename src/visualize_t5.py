#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 NMT Training Log Visualization Script
Extract loss and BLEU metrics from train.log and generate visualization charts
"""

import re
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
from pathlib import Path

# Configure matplotlib font settings
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

matplotlib.rcParams['axes.unicode_minus'] = False

def parse_log_file(log_path):
    """
    Parse training log file and extract loss and BLEU metrics
    
    Args:
        log_path: Path to the log file
        
    Returns:
        steps: List of training steps
        losses: List of loss values
        epochs: List of epoch numbers
        epoch_steps: List of step numbers at epoch evaluation
        epoch_losses: List of loss values at epoch end
        bleu1: List of BLEU-1 scores
        bleu2: List of BLEU-2 scores
        bleu3: List of BLEU-3 scores
        bleu4: List of BLEU-4 scores
    """
    steps = []
    losses = []
    epochs = []
    epoch_steps = []
    epoch_losses = []
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    
    # Regular expression patterns
    step_pattern = r'Step (\d+), Loss: ([\d.]+)'
    epoch_pattern = r'Epoch (\d+), Step (\d+), Loss: ([\d.]+).*?BLEU-1: ([\d.]+), BLEU-2: ([\d.]+), BLEU-3: ([\d.]+), BLEU-4: ([\d.]+)'
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match training step loss
            step_match = re.search(step_pattern, line)
            if step_match:
                step = int(step_match.group(1))
                loss = float(step_match.group(2))
                steps.append(step)
                losses.append(loss)
            
            # Match epoch evaluation results
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                step = int(epoch_match.group(2))
                loss = float(epoch_match.group(3))
                b1 = float(epoch_match.group(4))
                b2 = float(epoch_match.group(5))
                b3 = float(epoch_match.group(6))
                b4 = float(epoch_match.group(7))
                
                epochs.append(epoch)
                epoch_steps.append(step)
                epoch_losses.append(loss)
                bleu1.append(b1)
                bleu2.append(b2)
                bleu3.append(b3)
                bleu4.append(b4)
    
    return steps, losses, epochs, epoch_steps, epoch_losses, bleu1, bleu2, bleu3, bleu4

def visualize_training_curves(log_path, output_path=None):
    """
    Visualize training curves
    
    Args:
        log_path: Path to the log file
        output_path: Path to save the output image (optional)
    """
    print(f"Parsing log file: {log_path}")
    steps, losses, epochs, epoch_steps, epoch_losses, bleu1, bleu2, bleu3, bleu4 = parse_log_file(log_path)
    
    if not steps:
        print("Error: No data extracted from log file")
        return
    
    print(f"Extracted {len(steps)} training steps")
    print(f"Extracted {len(epochs)} epoch evaluation results")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss curve
    ax1.plot(steps, losses, 'b-', alpha=0.6, linewidth=0.8, label='Training Loss')
    if epoch_steps and epoch_losses:
        ax1.plot(epoch_steps, epoch_losses, 'ro', markersize=8, label='Epoch Eval Loss', zorder=5)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot BLEU curves
    if epochs and bleu1:
        ax2.plot(epochs, bleu1, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='BLEU-1')
        ax2.plot(epochs, bleu2, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='BLEU-2')
        ax2.plot(epochs, bleu3, '^-', color='#2ca02c', linewidth=2, markersize=6, label='BLEU-3')
        ax2.plot(epochs, bleu4, 'd-', color='#d62728', linewidth=2, markersize=6, label='BLEU-4')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU Score', fontsize=12)
        ax2.set_title('BLEU Evaluation Curve', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')
        ax2.set_xticks(epochs)
    else:
        ax2.text(0.5, 0.5, 'No BLEU evaluation data found', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('BLEU Evaluation Curve', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    else:
        # Default: save to the same directory as log file
        log_dir = os.path.dirname(log_path)
        output_path = os.path.join(log_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    
    # Display chart
    plt.show()
    
    # Print statistics
    if losses:
        print(f"\nTraining Loss Statistics:")
        print(f"  Min: {min(losses):.4f} (Step {steps[losses.index(min(losses))]})")
        print(f"  Max: {max(losses):.4f} (Step {steps[losses.index(max(losses))]})")
        print(f"  Mean: {sum(losses)/len(losses):.4f}")
        if epoch_losses:
            print(f"  Final Epoch Loss: {epoch_losses[-1]:.4f}")
    
    if bleu1:
        print(f"\nBLEU Score Statistics:")
        print(f"  BLEU-1: Best {max(bleu1):.4f} (Epoch {epochs[bleu1.index(max(bleu1))]}), Final {bleu1[-1]:.4f}")
        print(f"  BLEU-2: Best {max(bleu2):.4f} (Epoch {epochs[bleu2.index(max(bleu2))]}), Final {bleu2[-1]:.4f}")
        print(f"  BLEU-3: Best {max(bleu3):.4f} (Epoch {epochs[bleu3.index(max(bleu3))]}), Final {bleu3[-1]:.4f}")
        print(f"  BLEU-4: Best {max(bleu4):.4f} (Epoch {epochs[bleu4.index(max(bleu4))]}), Final {bleu4[-1]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='T5 NMT Training Log Visualization Tool')
    parser.add_argument('--log', type=str, 
                       default='../outputs/2025-12-28/23-19-24/train.log',
                       help='Path to training log file (relative to src directory)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (optional, default: same directory as log file)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Process log file path
    if os.path.isabs(args.log):
        log_path = args.log
    else:
        log_path = script_dir / args.log
    
    if not os.path.exists(log_path):
        print(f"Error: Log file does not exist: {log_path}")
        return
    
    # Process output path
    output_path = None
    if args.output:
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = script_dir / args.output
    
    visualize_training_curves(str(log_path), output_path)

if __name__ == '__main__':
    main()
