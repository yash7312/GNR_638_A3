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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import random

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

models_to_run = ["resnet50", "inception_v3", "efficientnet_b0"]

DATA_PATH = "train_data/train_data"
NUM_CLASSES = 30
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3

os.makedirs("checkpoints", exist_ok=True)

def get_linear_probe_model(model_name):

    model = timm.create_model(model_name, pretrained=True)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier (New Head)
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
        input_size = (1, 3, 299, 299)
    else:
        input_size = (1, 3, 224, 224)

    dummy_input = torch.randn(input_size).to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs

    print("\n============================================")
    print(f"Model: {model_name}")
    print(f"Total Parameters : {params:,}")
    print(f"MACs             : {macs/1e9:.3f} GMACs")
    print(f"FLOPs            : {flops/1e9:.3f} GFLOPs")
    print("============================================\n")


def get_dataloaders(model):

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = datasets.ImageFolder(DATA_PATH, transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader( train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion, epoch):

    model.train()
    correct, total = 0, 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

    for images, labels in progress_bar:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = correct / total

        progress_bar.set_postfix(acc=f"{acc:.3f}")

    return correct / total

def evaluate(model, loader, epoch):

    model.eval()

    correct, total = 0, 0

    progress_bar = tqdm(loader, desc=f"Val Epoch {epoch+1}", leave=False)

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

def compute_confusion(model, loader, model_name):

    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(xticks_rotation=90)

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(f"{model_name}_linear_probe_confusion_matrix.png")

    plt.close()

def extract_features(model, loader):

    model.eval()

    features = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)
            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)
            else:
                feats = model(images)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.reshape(feats.size(0), -1)
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(features), np.concatenate(labels_list)

def visualize_embeddings(features, labels, model_name):

    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(features)
    plt.figure()
    plt.scatter(reduced_pca[:,0], reduced_pca[:,1], c=labels, cmap="tab20", s=5)
    plt.title(f"{model_name} PCA")
    plt.savefig(f"{model_name}_linear_probe_pca.png")
    plt.close()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_tsne = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(reduced_tsne[:,0], reduced_tsne[:,1], c=labels, cmap="tab20", s=5)
    plt.title(f"{model_name} t-SNE")
    plt.savefig(f"{model_name}_linear_probe_tsne.png")
    plt.close()



for model_name in models_to_run:
    print(f"\nRunning Linear Probe for {model_name}")

    model = get_linear_probe_model(model_name)
    compute_efficiency(model, model_name)
    train_loader, val_loader = get_dataloaders(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    train_acc_history = []
    val_acc_history = []

    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_acc = evaluate(model, val_loader, epoch)
        epoch_time = time.time() - epoch_start
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc {train_acc:.3f} | "
            f"Val Acc {val_acc:.3f} | "
            f"Time {epoch_time:.0f}s"
        )

    total_time = time.time() - start_time
    print(f"\nTotal Training Time for {model_name}: {total_time/60:.2f} minutes")

    # Accuracy curve
    plt.figure()
    plt.plot(train_acc_history, label="Train")
    plt.plot(val_acc_history, label="Validation")
    plt.legend()
    plt.title(f"{model_name} Accuracy Curve")
    plt.savefig(f"{model_name}_linear_probe_accuracy_curve.png")
    plt.close()

    compute_confusion(model, val_loader, model_name)
    features, labels = extract_features(model, val_loader)
    visualize_embeddings(features, labels, model_name)
    torch.save(
        model.state_dict(),
        f"checkpoints/{model_name}_linear_probe.pth"
    )
