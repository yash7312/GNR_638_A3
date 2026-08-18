import torch
import torch.nn as nn
import timm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from thop import profile
import time
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants
SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["efficientnet_b0", "inception_v3", "resnet50"]

DATA_PATH = "train_data/train_data"

NUM_CLASSES = 30
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32

DATA_REGIMES = [1.0, 0.2, 0.05] # 100% 20% and 5%


def create_model(model_name):
    
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


def get_few_shot_loaders(model, subset_fraction):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    full_dataset = datasets.ImageFolder(DATA_PATH, transform)
    
    # 1. Split into Train and Val as 80 20
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(
        full_dataset, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # 2. Subsample based on the regime (100%, 20%, 5%)
    if subset_fraction < 1.0:
        keep_len = int(subset_fraction * len(train_ds))
        discard_len = len(train_ds) - keep_len
        train_ds, _ = random_split(
            train_ds, 
            [keep_len, discard_len], 
            generator=torch.Generator().manual_seed(SEED)
        )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    
    model.train()
    
    correct, total = 0, 0
    running_loss = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    
    for images, labels in progress_bar:
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        acc = correct / total
        progress_bar.set_postfix(
            acc=f"{acc:.3f}",
            loss=f"{running_loss/(total/BATCH_SIZE):.3f}"
        )
    
    return correct / total


def evaluate(model, loader):
    
    model.eval()
    
    correct, total = 0, 0
    
    with torch.no_grad():
        
        for images, labels in loader:
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def train_regime(model_name, fraction):
    print(f"\n===== {model_name} | Data Regime: {fraction*100:.0f}% =====")
    
    model = create_model(model_name)
    
    train_loader, val_loader = get_few_shot_loaders(model, fraction)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    final_train_acc = 0.0
    train_history = []
    val_history = []

    # Training loop
    for epoch in range(EPOCHS):

        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        final_train_acc = train_acc
        train_history.append(train_acc)
        
        val_acc = evaluate(model, val_loader)
        val_history.append(val_acc)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )
    return final_train_acc, best_val_acc, model, train_history, val_history


def compute_relative_drop(results, model_name):
    # Compute the relative drop in accuracy between 100% and 5% data regimes.
    acc_100 = results[model_name][1.0]["val"]
    acc_5 = results[model_name][0.05]["val"]
    
    delta = (acc_100 - acc_5) / acc_100
    
    return delta


def compute_train_val_gap(results, model_name, fraction):
    # Compute the train-validation gap
    train_acc = results[model_name][fraction]["train"]
    val_acc = results[model_name][fraction]["val"]
    
    return train_acc - val_acc


def print_results(results):
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for model_name in MODELS:
        delta = compute_relative_drop(results, model_name)
        
        print(f"\n{model_name}:")
        print(f"  Relative Drop (Delta): {delta:.4f}")
        
        for fraction in DATA_REGIMES:
            gap = compute_train_val_gap(results, model_name, fraction)
            val_acc = results[model_name][fraction]["val"]
            
            print(
                f"  Regime {fraction*100:3.0f}% | "
                f"Val Acc: {val_acc:.3f} | "
                f"Train-Val Gap: {gap:.3f}"
            )
    
    print("\n" + "="*60)


def run_experiments():
    print(f"Device: {device}")
    print(f"Models: {MODELS}")
    print(f"Data Regimes: {[f'{f*100:.0f}%' for f in DATA_REGIMES]}")

    base_results_dir = "few_shot_analysis"
    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(f"{base_results_dir}/plots", exist_ok=True)
    os.makedirs(f"{base_results_dir}/efficiency_metrics", exist_ok=True)
    os.makedirs(f"{base_results_dir}/overfitting_analysis", exist_ok=True)
    
    # Initialize results storage
    results = {model_name: {} for model_name in MODELS}
    efficiency_results = {}
    overfitting_results = {model_name: {} for model_name in MODELS}
    
    # Train each model on each data regime
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Starting experiments for {model_name}")
        print(f"{'='*60}")

        model_efficiency = compute_efficiency_metrics(create_model(model_name), model_name)
        efficiency_results[model_name] = model_efficiency
        
        for fraction in DATA_REGIMES:
            train_acc, val_acc, trained_model, train_history, val_history = train_regime(model_name, fraction)
            results[model_name][fraction] = {
                "train": train_acc, 
                "val": val_acc,
                "train_history": train_history,
                "val_history": val_history
            }

            overfitting_results[model_name][fraction] = compute_overfitting_score(
                train_acc, val_acc, train_history, val_history
            )

            del trained_model
        plot_learning_curves(results, model_name, save_dir=f"{base_results_dir}/plots")
    
    print_results(results)

    save_analysis_results(
        results,
        efficiency_results,
        overfitting_results,
        base_results_dir
    )
    
    return results

# efficiency analysis for bonus
def compute_efficiency_metrics(model, model_name):
    """Tracking FLOPs, parameters, and memory usage"""
    model.eval()
    input_size = (1, 3, 299, 299) if model_name == "inception_v3" else (1, 3, 224, 224)
    dummy = torch.randn(input_size).to(device)
    
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": params,
        "trainable_params": trainable,
        "flops": flops,
        "trainable_ratio": trainable / params
    }

def compute_overfitting_score(train_acc, val_acc, train_history, val_history):
    """Quantify overfitting severity"""
    gap = train_acc - val_acc
    trend = np.polyfit(range(len(val_history)), val_history, 1)[0]  # Slope
    
    return {
        "gap": gap,
        "val_trend": trend,
        "severity": "High" if gap > 0.15 else "Medium" if gap > 0.08 else "Low"
    }

def plot_learning_curves(results, model_name, save_dir="few_shot_analysis"):
    """Create detailed learning curve visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Accuracy vs Data Regime
    regimes = [f"{f*100:.0f}%" for f in DATA_REGIMES]
    val_accs = [results[model_name][f]["val"] for f in DATA_REGIMES]
    train_accs = [results[model_name][f]["train"] for f in DATA_REGIMES]
    
    axes[0, 0].plot(regimes, val_accs, 'o-', label='Val Acc')
    axes[0, 0].plot(regimes, train_accs, 's-', label='Train Acc')
    axes[0, 0].set_title(f'{model_name} - Accuracy vs Data')
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Overfitting Gap
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    colors = ['red' if g > 0.15 else 'orange' if g > 0.08 else 'green' for g in gaps]
    axes[0, 1].bar(regimes, gaps, color=colors, alpha=0.7)
    axes[0, 1].set_title('Train-Val Gap (Overfitting Severity)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Gap')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Training curves for 100% data regime
    if 1.0 in results[model_name]:
        train_hist = results[model_name][1.0]["train_history"]
        val_hist = results[model_name][1.0]["val_history"]
        epochs = range(1, len(train_hist) + 1)
        axes[1, 0].plot(epochs, train_hist, 'o-', label='Train', linewidth=2, markersize=6)
        axes[1, 0].plot(epochs, val_hist, 's-', label='Val', linewidth=2, markersize=6)
        axes[1, 0].set_title(f'{model_name} - Learning Curves (100% Data)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Sample efficiency
    if 1.0 in results[model_name]:
        regimes_numeric = [f*100 for f in DATA_REGIMES]
        val_accs_list = [results[model_name][f]["val"] for f in DATA_REGIMES]
        axes[1, 1].plot(regimes_numeric, val_accs_list, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_title(f'{model_name} - Data Efficiency', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Data Regime (%)')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_analysis.png", dpi=300)
    plt.close()
    print(f"Saved plot: {save_dir}/{model_name}_analysis.png")
    
def save_analysis_results(results, efficiency_results, overfitting_results, base_dir):
    """Save all analysis results to CSV and TXT files"""
    
    # 1. Save Efficiency Metrics to CSV
    efficiency_data = []
    for model_name in MODELS:
        metrics = efficiency_results[model_name]
        efficiency_data.append({
            'Model': model_name,
            'Total Parameters': metrics['total_params'],
            'Trainable Parameters': metrics['trainable_params'],
            'FLOPs': metrics['flops'],
            'Trainable Ratio': metrics['trainable_ratio']
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    efficiency_df.to_csv(f"{base_dir}/efficiency_metrics/model_efficiency.csv", index=False)
    print(f"\nSaved efficiency metrics to {base_dir}/efficiency_metrics/model_efficiency.csv")
    
    # 2. Save Overfitting Analysis to CSV
    overfit_data = []
    for model_name in MODELS:
        for fraction, overfit_score in overfitting_results[model_name].items():
            overfit_data.append({
                'Model': model_name,
                'Data Regime': f"{fraction*100:.0f}%",
                'Train-Val Gap': overfit_score['gap'],
                'Validation Trend': overfit_score['val_trend'],
                'Severity': overfit_score['severity']
            })
    
    overfit_df = pd.DataFrame(overfit_data)
    overfit_df.to_csv(f"{base_dir}/overfitting_analysis/overfitting_metrics.csv", index=False)
    print(f"Saved overfitting analysis to {base_dir}/overfitting_analysis/overfitting_metrics.csv")
    
    # 3. Save comprehensive results summary to TXT
    with open(f"{base_dir}/comprehensive_results.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEW-SHOT LEARNING ANALYSIS - COMPREHENSIVE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Model Efficiency Summary
        f.write("MODEL EFFICIENCY METRICS\n")
        f.write("-"*80 + "\n")
        for model_name in MODELS:
            metrics = efficiency_results[model_name]
            f.write(f"\n{model_name}:\n")
            f.write(f"  Total Parameters: {metrics['total_params']:,}\n")
            f.write(f"  Trainable Parameters: {metrics['trainable_params']:,}\n")
            f.write(f"  FLOPs: {metrics['flops']:,.0f}\n")
            f.write(f"  Trainable Ratio: {metrics['trainable_ratio']:.4f}\n")
        
        # Accuracy Results
        f.write("\n\n" + "="*80 + "\n")
        f.write("ACCURACY RESULTS BY DATA REGIME\n")
        f.write("="*80 + "\n")
        for model_name in MODELS:
            delta = compute_relative_drop(results, model_name)
            f.write(f"\n{model_name}:\n")
            f.write(f"  Relative Drop (100% to 5%): {delta:.4f}\n")
            f.write(f"  {'-'*76}\n")
            f.write(f"  {'Regime':<12} {'Train Acc':<14} {'Val Acc':<14} {'Train-Val Gap':<14}\n")
            f.write(f"  {'-'*76}\n")
            
            for fraction in DATA_REGIMES:
                train_acc = results[model_name][fraction]['train']
                val_acc = results[model_name][fraction]['val']
                gap = train_acc - val_acc
                f.write(f"  {fraction*100:3.0f}%{'':<8} {train_acc:<14.4f} {val_acc:<14.4f} {gap:<14.4f}\n")
        
        # Overfitting Analysis
        f.write("\n\n" + "="*80 + "\n")
        f.write("OVERFITTING ANALYSIS\n")
        f.write("="*80 + "\n")
        for model_name in MODELS:
            f.write(f"\n{model_name}:\n")
            f.write(f"  {'Data Regime':<14} {'Gap':<12} {'Val Trend':<14} {'Severity':<12}\n")
            f.write(f"  {'-'*52}\n")
            for fraction in DATA_REGIMES:
                score = overfitting_results[model_name][fraction]
                f.write(f"  {fraction*100:3.0f}%{'':<10} {score['gap']:<12.4f} {score['val_trend']:<14.6f} {score['severity']:<12}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Saved comprehensive results to {base_dir}/comprehensive_results.txt")

if __name__ == "__main__":
    results = run_experiments()
    
    