import torch
import torch.nn as nn
import timm
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from thop import profile

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


DATA_PATH = "train_data/train_data"
NUM_CLASSES = 30
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

MODELS = ["efficientnet_b0","resnet50","inception_v3"]

os.makedirs("probe_plots",exist_ok=True)

def get_dataloaders(model):

    config = resolve_data_config({},model=model)
    transform = create_transform(**config)

    dataset = datasets.ImageFolder(DATA_PATH,transform)

    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size

    train_ds,val_ds = random_split(dataset,[train_size,val_size],generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

    return train_loader,val_loader,dataset

def create_model(model_name):

    model = timm.create_model(model_name,pretrained=True)
    if model_name=="resnet50":

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features,NUM_CLASSES)

    elif model_name=="efficientnet_b0":

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features,NUM_CLASSES)

    elif model_name=="inception_v3":

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features,NUM_CLASSES)

    return model.to(device)

def load_checkpoint(model,model_name):

    path = f"checkpoints_ft/{model_name}_FullFT.pth"
    if os.path.exists(path):
        print(f"Loading checkpoint: {path}")
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    else:
        print("WARNING: checkpoint not found, using pretrained model")

def compute_efficiency(model,model_name):
    model.eval()
    if model_name=="inception_v3":
        input_size=(1,3,299,299)
    else:
        input_size=(1,3,224,224)
    dummy=torch.randn(input_size).to(device)
    macs,params=profile(model,inputs=(dummy,),verbose=False)
    flops=2*macs

    print("\n======================================")
    print(f"{model_name}")
    print(f"Parameters : {params:,}")
    print(f"MACs       : {macs/1e9:.3f} GMACs")
    print(f"FLOPs      : {flops/1e9:.3f} GFLOPs")
    print("======================================\n")

def register_hooks(model,model_name):
    features={}
    def hook(name):
        def fn(module,input,output):
            features[name]=output.detach()
        return fn

    if model_name=="resnet50":
        model.layer1.register_forward_hook(hook("early"))
        model.layer3.register_forward_hook(hook("middle"))
        model.layer4.register_forward_hook(hook("final"))

    elif model_name=="efficientnet_b0":
        model.blocks[1].register_forward_hook(hook("early"))
        model.blocks[4].register_forward_hook(hook("middle"))
        model.blocks[-1].register_forward_hook(hook("final"))

    elif model_name=="inception_v3":
        model.Mixed_5d.register_forward_hook(hook("early"))
        model.Mixed_6e.register_forward_hook(hook("middle"))
        model.Mixed_7c.register_forward_hook(hook("final"))

    return features

def extract_features(model,loader,features):
    X={"early":[],"middle":[],"final":[]}
    y=[]
    model.eval()

    with torch.no_grad():
        for images,labels in tqdm(loader,desc="Extracting Features",leave=False):
            images=images.to(device)
            model(images)
            for k in X:
                feat=features[k]
                feat=torch.mean(feat,dim=(2,3))
                X[k].append(feat.cpu())
            y.append(labels)

    for k in X:
        X[k]=torch.cat(X[k])

    y=torch.cat(y)
    return X,y

def train_probe(X_train,y_train,X_val,y_val):
    in_dim=X_train.shape[1]
    clf=nn.Linear(in_dim,NUM_CLASSES).to(device)
    optimizer=torch.optim.Adam(clf.parameters(),lr=LR)
    criterion=nn.CrossEntropyLoss()
    X_train=X_train.to(device)
    y_train=y_train.to(device)
    X_val=X_val.to(device)
    y_val=y_val.to(device)

    for epoch in tqdm(range(EPOCHS),desc="Training Linear Probe",leave=False):
        clf.train()
        logits=clf(X_train)
        loss=criterion(logits,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        preds=clf(X_val).argmax(1)
        acc=(preds==y_val).float().mean().item()

    return acc

def get_pca_subset(dataset):
    targets=np.array(dataset.targets)
    selected=[]
    for c in range(NUM_CLASSES):
        idx=np.where(targets==c)[0][:30]
        selected.extend(idx)

    return Subset(dataset,selected)

print("\nCreating fixed PCA subset")
dummy_model = timm.create_model("resnet50", pretrained=True)
config = resolve_data_config({}, model=dummy_model)
transform = create_transform(**config)
dataset_global = datasets.ImageFolder(DATA_PATH, transform=transform)
pca_subset = get_pca_subset(dataset_global)


for model_name in MODELS:
    print("\n======================================")
    print(f"Layer-wise probing: {model_name}")
    print("======================================")

    model=create_model(model_name)
    load_checkpoint(model,model_name)
    compute_efficiency(model,model_name)
    train_loader,val_loader,_=get_dataloaders(model)
    features=register_hooks(model,model_name)

    print("Extracting Train Features")
    X_train,y_train=extract_features(model,train_loader,features)

    print("Extracting Validation Features")
    X_val,y_val=extract_features(model,val_loader,features)

    depths=["early","middle","final"]

    accs=[]
    norms=[]

    for d in depths:
        print(f"\nTraining probe for {d}")
        acc=train_probe(X_train[d],y_train,X_val[d],y_val)
        accs.append(acc)
        norm=torch.norm(X_train[d],dim=1).mean().item()
        norms.append(norm)

    plt.figure()
    plt.plot(depths,accs,marker="o")
    plt.title(f"{model_name} Validation Accuracy vs Depth")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Layer Depth")
    plt.savefig(f"probe_plots/{model_name}_probe_accuracy.png",dpi=300)
    plt.close()

    plt.figure()
    plt.bar(depths,norms)
    plt.title(f"{model_name} Feature Norms")
    plt.ylabel("Average Feature Norm")
    plt.savefig(f"probe_plots/{model_name}_feature_norms.png",dpi=300)
    plt.close()

    subset_loader=DataLoader(pca_subset,batch_size=64,shuffle=False)
    X_subset,y_subset=extract_features(model,subset_loader,features)
    
    for d in depths:
        pca=PCA(n_components=2)
        proj=pca.fit_transform(X_subset[d])
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(proj[:,0],proj[:,1],c=y_subset,cmap="nipy_spectral",s=12,alpha=0.85)
        plt.title(f"{model_name} PCA {d}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter,label="Class")
        plt.savefig(f"probe_plots/{model_name}_pca_{d}.png",dpi=300)
        plt.close()