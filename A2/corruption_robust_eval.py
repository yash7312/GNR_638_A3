import torch
import torch.nn as nn
import timm
import random
import numpy as np
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from thop import profile
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import os
import csv
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

DATA_PATH = "train_data/train_data"
NUM_CLASSES = 30
BATCH_SIZE = 32
MODELS = ["efficientnet_b0", "resnet50", "inception_v3"]
OUTPUT_DIR = "robustness_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

results = []
efficiency_metrics = {}
all_model_results = {}

def create_model(model_name):
    model = timm.create_model(model_name, pretrained=False)

    if model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "efficientnet_b0":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "inception_v3":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(device)

def compute_efficiency(model, model_name):
    model.eval()
    if model_name == "inception_v3":
        input_size = (1,3,299,299)
    else:
        input_size = (1,3,224,224)

    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    metrics = {
        "params": params,
        "macs": macs,
        "flops": flops,
        "params_M": params / 1e6,
        "gflops": flops / 1e9
    }
    efficiency_metrics[model_name] = metrics

    print("\n==========================================")
    print(f"Efficiency Metrics for {model_name}")
    print(f"Parameters : {params:,} ({metrics['params_M']:.2f}M)")
    print(f"MACs       : {macs/1e9:.3f} GMACs")
    print(f"FLOPs      : {flops/1e9:.3f} GFLOPs")
    print("==========================================\n")

    return metrics

def gaussian_noise(img, sigma):
    tensor = TF.to_tensor(img)
    noise = torch.randn_like(tensor) * sigma
    tensor = torch.clamp(tensor + noise,0,1)

    return TF.to_pil_image(tensor)


def motion_blur(img, radius=5):
    return img.filter(ImageFilter.GaussianBlur(radius))


def brightness_shift(img, factor=1.5):
    return TF.adjust_brightness(img,factor)

def get_loader(model, corruption=None):
    config = resolve_data_config({}, model=model)
    base_transform = create_transform(**config)
    transform_list = []
    if corruption is not None:
        transform_list.append(corruption)

    transform_list.append(base_transform)
    transform = torchvision.transforms.Compose(transform_list)
    dataset = datasets.ImageFolder(DATA_PATH, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset,[train_size,val_size],generator=torch.Generator().manual_seed(42))

    loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)
    return loader

def evaluate(model, loader):
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for images,labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if isinstance(outputs,tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs,1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    return correct/total

def robustness_test(model_name):
    print("\n========================================")
    print(f"Robustness Evaluation for {model_name}")
    print("========================================")

    model = create_model(model_name)
    model.load_state_dict(torch.load(f"checkpoints_ft/{model_name}_FullFT.pth",map_location=device,weights_only=True))
    model.to(device)

    efficiency = compute_efficiency(model,model_name)
    clean_loader = get_loader(model)
    acc_clean = evaluate(model,clean_loader)
    print(f"\nClean Accuracy : {acc_clean:.4f}\n")

    corruptions = {
        "Gaussian_0.05": lambda img: gaussian_noise(img,0.05),
        "Gaussian_0.1": lambda img: gaussian_noise(img,0.1),
        "Gaussian_0.2": lambda img: gaussian_noise(img,0.2),
        "MotionBlur": lambda img: motion_blur(img,5),
        "BrightnessShift": lambda img: brightness_shift(img,1.5)
    }

    print(f"{'Corruption':20s} {'Accuracy':10s} {'Error':10s} {'RelRobust':10s}")

    model_corrupted_results = {}
    gaussian_accs = []

    for name,corr in corruptions.items():
        loader = get_loader(model,corr)
        acc = evaluate(model,loader)
        corruption_error = 1 - acc
        relative_robustness = acc / acc_clean
        accuracy_drop = acc_clean - acc

        print(
            f"{name:20s} "
            f"{acc:.4f}     "
            f"{corruption_error:.4f}     "
            f"{relative_robustness:.4f}"
        )

        results.append([model_name,acc_clean,name,acc,corruption_error,relative_robustness,accuracy_drop,efficiency['params_M'],efficiency['gflops']])

        model_corrupted_results[name] = {
            "accuracy":acc,
            "error":corruption_error,
            "relative_robustness":relative_robustness,
            "accuracy_drop":accuracy_drop
        }

        if "Gaussian" in name:
            gaussian_accs.append(acc)

    all_model_results[model_name] = {
        "clean_acc":acc_clean,
        "corruptions":model_corrupted_results,
        "gaussian_accs":gaussian_accs,
        "efficiency":efficiency
    }

def save_results_to_csv():
    csv_path = os.path.join(OUTPUT_DIR,"robustness_results.csv")
    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model","Clean_Accuracy","Corruption","Corrupted_Accuracy","Corruption_Error","Relative_Robustness","Accuracy_Drop","Params_M","GFLOPs"])
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")

def plot_robustness_curves():
    plt.figure(figsize=(10,6))
    sigmas=[0.05,0.1,0.2]
    for model_name,data in all_model_results.items():
        accs=[data["clean_acc"]]+data["gaussian_accs"]
        x=[0]+sigmas
        plt.plot(x,accs,marker="o",linewidth=2,label=model_name)

    plt.xlabel("Gaussian Noise Sigma")
    plt.ylabel("Validation Accuracy")
    plt.title("Robustness to Gaussian Noise")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(PLOTS_DIR,"gaussian_robustness_curves.png"),dpi=300)
    plt.close()

def plot_accuracy_drop():

    corruption_types=[
        "Gaussian_0.05",
        "Gaussian_0.1",
        "Gaussian_0.2",
        "MotionBlur",
        "BrightnessShift"
    ]

    fig,ax=plt.subplots(figsize=(12,6))

    x=np.arange(len(corruption_types))
    width=0.25

    for i,model in enumerate(MODELS):

        drops=[
            all_model_results[model]["corruptions"][c]["accuracy_drop"]
            for c in corruption_types
        ]

        ax.bar(x+i*width,drops,width,label=model)

    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Accuracy Drop Under Corruptions")

    ax.set_xticks(x+width)
    ax.set_xticklabels(corruption_types)

    ax.legend()

    plt.savefig(os.path.join(PLOTS_DIR,"accuracy_drop_plot.png"),dpi=300)
    plt.close()

def plot_corruption_comparison():
    corruption_types = [
        "Gaussian_0.05",
        "Gaussian_0.1",
        "Gaussian_0.2",
        "MotionBlur",
        "BrightnessShift"
    ]

    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(corruption_types))
    width = 0.25

    for i, model_name in enumerate(MODELS):
        accs = [
            all_model_results[model_name]["corruptions"][c]["accuracy"]
            for c in corruption_types
        ]
        ax.bar(x + i*width, accs, width, label=model_name)

    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Under Different Corruptions")
    ax.set_xticks(x + width)
    ax.set_xticklabels(corruption_types, rotation=15)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR,"corruption_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Corruption comparison plot saved to: {save_path}")

def plot_relative_robustness_heatmap():
    corruption_types = [
        "Gaussian_0.05",
        "Gaussian_0.1",
        "Gaussian_0.2",
        "MotionBlur",
        "BrightnessShift"
    ]

    data_matrix = []
    for model_name in MODELS:
        row = [
            all_model_results[model_name]["corruptions"][c]["relative_robustness"]
            for c in corruption_types
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(corruption_types)))
    ax.set_yticks(np.arange(len(MODELS)))

    ax.set_xticklabels(corruption_types, rotation=20)
    ax.set_yticklabels(MODELS)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Robustness")

    for i in range(len(MODELS)):
        for j in range(len(corruption_types)):
            ax.text(j,i,f"{data_matrix[i,j]:.3f}",ha="center",va="center",color="black")

    ax.set_title("Relative Robustness Heatmap")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR,"relative_robustness_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def generate_analysis_report():
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    corruption_types = [
        "Gaussian_0.05",
        "Gaussian_0.1",
        "Gaussian_0.2",
        "MotionBlur",
        "BrightnessShift"
    ]

    # Average relative robustness
    avg_relative_robustness = {}
    for model in MODELS:
        avg = np.mean([
            all_model_results[model]["corruptions"][c]["relative_robustness"]
            for c in corruption_types
        ])
        avg_relative_robustness[model] = avg

    most_efficient_model = min(
        efficiency_metrics,
        key=lambda x: efficiency_metrics[x]["params_M"]
    )

    with open(report_path, "w") as f:

        f.write("="*70 + "\n")
        f.write("CORRUPTION ROBUSTNESS ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("1. SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n\n")

        for model in MODELS:

            clean = all_model_results[model]["clean_acc"]
            params = efficiency_metrics[model]["params_M"]
            gflops = efficiency_metrics[model]["gflops"]

            f.write(f"{model}\n")
            f.write(f"  Clean Accuracy: {clean:.4f}\n")
            f.write(f"  Parameters: {params:.2f}M\n")
            f.write(f"  GFLOPs: {gflops:.2f}\n")
            f.write(f"  Avg Relative Robustness: {avg_relative_robustness[model]:.4f}\n\n")

        f.write("2. MODEL RANKINGS\n")
        f.write("-"*70 + "\n\n")

        sorted_clean = sorted(
            MODELS,
            key=lambda x: all_model_results[x]["clean_acc"],
            reverse=True
        )

        f.write("Ranking by Clean Accuracy:\n")

        for i, model in enumerate(sorted_clean,1):
            f.write(f"{i}. {model} ({all_model_results[model]['clean_acc']:.4f})\n")

        f.write("\nRanking by Robustness:\n")

        sorted_robust = sorted(
            avg_relative_robustness,
            key=avg_relative_robustness.get,
            reverse=True
        )

        for i, model in enumerate(sorted_robust,1):
            f.write(f"{i}. {model} ({avg_relative_robustness[model]:.4f})\n")

        f.write("\n3. ROBUSTNESS PER CORRUPTION\n")
        f.write("-"*70 + "\n\n")

        for corr in corruption_types:

            accs = {
                m: all_model_results[m]["corruptions"][corr]["accuracy"]
                for m in MODELS
            }

            best_model = max(accs, key=accs.get)
            worst_model = min(accs, key=accs.get)

            f.write(f"{corr}\n")
            f.write(f"  Best Model: {best_model} ({accs[best_model]:.4f})\n")
            f.write(f"  Worst Model: {worst_model} ({accs[worst_model]:.4f})\n\n")

        f.write("4. CORRUPTION DIFFICULTY\n")
        f.write("-"*70 + "\n\n")

        avg_acc = {}

        for corr in corruption_types:

            avg = np.mean([
                all_model_results[m]["corruptions"][corr]["accuracy"]
                for m in MODELS
            ])

            avg_acc[corr] = avg

        hardest = min(avg_acc, key=avg_acc.get)
        easiest = max(avg_acc, key=avg_acc.get)

        f.write(f"Hardest Corruption: {hardest} ({avg_acc[hardest]:.4f})\n")
        f.write(f"Easiest Corruption: {easiest} ({avg_acc[easiest]:.4f})\n\n")

        f.write("5. EFFICIENCY ANALYSIS\n")
        f.write("-"*70 + "\n\n")

        for model in MODELS:

            eff = efficiency_metrics[model]

            f.write(f"{model}\n")
            f.write(f"  Parameters: {eff['params_M']:.2f}M\n")
            f.write(f"  GFLOPs: {eff['gflops']:.2f}\n\n")

        f.write(f"Most Efficient Model: {most_efficient_model}\n\n")

        f.write("6. RECOMMENDATIONS\n")
        f.write("-"*70 + "\n\n")

        # Best model for Gaussian noise
        noise_acc = {
            m: np.mean([
                all_model_results[m]["corruptions"]["Gaussian_0.05"]["accuracy"],
                all_model_results[m]["corruptions"]["Gaussian_0.1"]["accuracy"],
                all_model_results[m]["corruptions"]["Gaussian_0.2"]["accuracy"]
            ])
            for m in MODELS
        }

        best_noise_model = max(noise_acc, key=noise_acc.get)

        # Best model for motion blur
        blur_acc = {
            m: all_model_results[m]["corruptions"]["MotionBlur"]["accuracy"]
            for m in MODELS
        }

        best_blur_model = max(blur_acc, key=blur_acc.get)

        # Best model for brightness shift
        brightness_acc = {
            m: all_model_results[m]["corruptions"]["BrightnessShift"]["accuracy"]
            for m in MODELS
        }

        best_brightness_model = max(brightness_acc, key=brightness_acc.get)

        # Most efficient model
        least_params_model = min(
            efficiency_metrics,
            key=lambda x: efficiency_metrics[x]["params_M"]
        )

        least_flops_model = min(
            efficiency_metrics,
            key=lambda x: efficiency_metrics[x]["gflops"]
        )

        # Overall robust model
        best_overall = max(avg_relative_robustness, key=avg_relative_robustness.get)

        f.write(f"For noisy environments: use {best_noise_model}\n")
        f.write(f"For motion blur scenarios (moving cameras / moving objects): use {best_blur_model}\n")
        f.write(f"For illumination variations (brightness changes): use {best_brightness_model}\n")
        f.write(f"Most memory efficient: use {least_params_model}\n")
        f.write(f"Most compute efficient: use {least_flops_model}\n")
        f.write(f"For balanced performance across all corruptions: use {best_overall}\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")

    print(f"\nAnalysis report saved to: {report_path}")


if __name__ == "__main__":

    print("\n" + "="*70)
    print("TASK 4.4 ROBUSTNESS ANALYSIS")
    print("="*70)

    for model_name in MODELS:
        robustness_test(model_name)

    save_results_to_csv()

    plot_robustness_curves()
    plot_accuracy_drop()
    plot_corruption_comparison()
    plot_relative_robustness_heatmap()
    generate_analysis_report()

    print("\nEvaluation Complete.")
