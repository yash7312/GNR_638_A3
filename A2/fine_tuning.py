import torch
import torch.nn as nn
import timm
import time
from thop import profile
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["efficientnet_b0", "inception_v3", "resnet50"]
DATA_PATH = "train_data/train_data"
NUM_CLASSES = 30
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32

os.makedirs("checkpoints_ft", exist_ok=True)


def compute_efficiency(model, model_name):

    model.eval()
    if model_name == "inception_v3":
        input_size = (1,3,299,299)
    else:
        input_size = (1,3,224,224)

    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs

    print("\n================================================")
    print(f"Efficiency Metrics for {model_name}")
    print(f"Parameters : {params:,}")
    print(f"MACs       : {macs/1e9:.3f} GMACs")
    print(f"FLOPs      : {flops/1e9:.3f} GFLOPs")
    print("================================================\n")


def get_dataloaders(model):

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = datasets.ImageFolder(DATA_PATH, transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def create_model(model_name):

    model = timm.create_model(model_name, pretrained=True)

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


def linear_probe(model):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "fc" in name or "classifier" in name:
            p.requires_grad = True


def full_finetune(model):
    for p in model.parameters():
        p.requires_grad = True


def last_block_finetune(model):

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "layer4"):
        for p in model.layer4.parameters():
            p.requires_grad = True

    elif hasattr(model, "blocks"):
        for p in model.blocks[-1].parameters():
            p.requires_grad = True

    elif hasattr(model, "Mixed_7c"):
        for p in model.Mixed_7c.parameters():
            p.requires_grad = True


def selective_20_percent(model):

    for p in model.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    target = int(0.2 * total)

    count = 0
    for p in reversed(list(model.parameters())):
        if count >= target:
            break

        p.requires_grad = True
        count += p.numel()


def get_unfrozen_percentage(model):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return 100 * trainable / total


def train_one_epoch(model, loader, optimizer, criterion, epoch):

    model.train()
    correct, total = 0, 0
    running_loss = 0
    grad_epoch = 0

    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)

    for images, labels in progress_bar:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        grad_epoch += total_norm ** 0.5

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

    return correct / total, running_loss, grad_epoch


def evaluate(model, loader, epoch):

    model.eval()

    correct, total = 0, 0

    progress_bar = tqdm(loader, desc=f"Validation Epoch {epoch+1}", leave=False)

    with torch.no_grad():

        for images, labels in progress_bar:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            progress_bar.set_postfix(val_acc=f"{acc:.3f}")

    return correct / total


def train_strategy(model_name, strategy_fn, strategy_name):

    print(f"\n===== {model_name} | {strategy_name} =====")

    model = create_model(model_name)

    strategy_fn(model)

    percent = get_unfrozen_percentage(model)

    print(f"Trainable Parameters: {percent:.2f}%")

    train_loader, val_loader = get_dataloaders(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    train_acc_hist = []
    val_acc_hist = []
    loss_hist = []
    grad_norm_hist = []

    for epoch in range(EPOCHS):

        train_acc, loss, grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )

        val_acc = evaluate(model, val_loader, epoch)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        loss_hist.append(loss)
        grad_norm_hist.append(grad_norm)

        print(
            f"Epoch {epoch+1} | "
            f"Train {train_acc:.3f} | "
            f"Val {val_acc:.3f} | "
            f"GradNorm {grad_norm:.2f}"
        )

    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Validation")
    plt.legend()
    plt.title(f"{model_name}-{strategy_name} Accuracy")
    plt.savefig(f"{model_name}_{strategy_name}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"{model_name}-{strategy_name} Loss")
    plt.savefig(f"{model_name}_{strategy_name}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(grad_norm_hist)
    plt.title(f"{model_name}-{strategy_name} Gradient Norm")
    plt.savefig(f"{model_name}_{strategy_name}_gradnorm.png")
    plt.close()

    torch.save(
        model.state_dict(),
        f"checkpoints_ft/{model_name}_{strategy_name}.pth"
    )

    return percent, train_acc, val_acc


strategies = {

"LinearProbe": linear_probe,
"LastBlockFT": last_block_finetune,
"FullFT": full_finetune,
"Selective20": selective_20_percent

}


for model_name in MODELS:

    print("\n=====================================")
    print(f"Starting experiments for {model_name}")
    print("=====================================")

    base_model = create_model(model_name)

    compute_efficiency(base_model, model_name)

    train_acc_vs_unfreeze = []
    val_acc_vs_unfreeze = []

    start_time = time.time()

    for strategy_name, fn in strategies.items():

        percent, train_acc, val_acc = train_strategy(
            model_name,
            fn,
            strategy_name
        )

        train_acc_vs_unfreeze.append((percent, train_acc))
        val_acc_vs_unfreeze.append((percent, val_acc))

    end_time = time.time()

    total_time = end_time - start_time

    print(f"\nTotal Training Time for {model_name}: {total_time/60:.2f} minutes")

    # Training accuracy plot
    x = [a[0] for a in train_acc_vs_unfreeze]
    y = [a[1] for a in train_acc_vs_unfreeze]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("% Unfrozen Parameters")
    plt.ylabel("Training Accuracy")
    plt.title(f"{model_name} Training Accuracy vs Unfrozen Params")
    plt.savefig(f"{model_name}_train_accuracy_vs_unfreeze.png")
    plt.close()

    # Validation accuracy plot
    x = [a[0] for a in val_acc_vs_unfreeze]
    y = [a[1] for a in val_acc_vs_unfreeze]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("% Unfrozen Parameters")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{model_name} Validation Accuracy vs Unfrozen Params")
    plt.savefig(f"{model_name}_val_accuracy_vs_unfreeze.png")
    plt.close()
