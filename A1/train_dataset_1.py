import custom_core_dataset_1 as core
import os, time, math, random, sys
import pickle
import matplotlib.pyplot as plt

random.seed(42)

def train():
    SOURCE_PATH = os.path.abspath(input("Give the path to dataset: "))
    print(f"The path provided is {SOURCE_PATH}")
    print("Loading Data...")
    start_time = time.time()
    imgs, labels = core.load_dataset_1(SOURCE_PATH)
    end_time = time.time()
    print(f"Total Laoding Time: {end_time - start_time:.2f} seconds")
    data = list(zip(imgs, labels))
    random.shuffle(data)
    train_data = data

    C_OUT = 8
    LR = 0.01
    EPOCHS = 10
    MOM, WD = 0.9, 1e-4

    print_model_stats(C_OUT,NUM_CLASSES=10)

    c1_w = core.Tensor([C_OUT, 3, 3, 3])
    c1_b = core.Tensor([C_OUT])
    l1_w = core.Tensor([10, C_OUT*16*16])
    l1_b = core.Tensor([10])

    # He init
    for p in [c1_w, l1_w]:
        if len(p.shape) == 4:
            fan_in = p.shape[1] * p.shape[2] * p.shape[3]
        else:
            fan_in = p.shape[1]
        std = math.sqrt(2.0 / fan_in)
        p.data = [random.gauss(0, std) for _ in p.data]

    act_conv = core.Tensor([1, C_OUT, 32, 32])
    act_pool = core.Tensor([1, C_OUT, 16, 16])
    pool_mask = core.Tensor([1, C_OUT, 32, 32])
    act_flat = core.Tensor([1, C_OUT*16*16])
    act_logits = core.Tensor([1, 10])

    grad_conv = core.Tensor([1, C_OUT, 32, 32])
    grad_pool = core.Tensor([1, C_OUT, 16, 16])

    epoch_losses = []
    epoch_accs = []

    for ep in range(EPOCHS):
        random.shuffle(train_data)
        correct = 0
        loss_sum = 0
        t0 = time.time()

        for img, y in train_data:
            for t in [c1_w, c1_b, l1_w, l1_b,
                      act_conv, act_pool, act_flat,
                      act_logits, grad_conv, grad_pool, img]:
                t.zero_grad()

            core.conv2d_fwd(img, c1_w, c1_b, act_conv, 1, 1)
            core.relu_fwd(act_conv)
            core.maxpool_fwd(act_conv, act_pool, pool_mask, 2, 2)
            core.flatten(act_pool, act_flat)
            core.linear_fwd(act_flat, l1_w, l1_b, act_logits)

            loss_sum += core.softmax_cross_entropy_grad(act_logits, y)
            if act_logits.data.index(max(act_logits.data)) == y:
                correct += 1

            core.linear_bwd(act_flat, l1_w, l1_b, act_logits)
            core.maxpool_bwd(act_logits, grad_pool, pool_mask)
            core.relu_bwd(act_conv, grad_conv)
            core.conv2d_bwd(img, c1_w, c1_b, grad_conv, 1, 1)

            for p in [c1_w, c1_b, l1_w, l1_b]:
                core.sgd_momentum_step(p, LR, MOM, WD)

        avg_loss = loss_sum / len(train_data)
        acc = correct / len(train_data)

        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)

        print(f"Epoch {ep+1} | Loss {loss_sum/len(train_data):.4f} "
              f"| Acc {correct/len(train_data):.2%} "
              f"| Time {time.time()-t0:.1f}s")

        LR *= 0.8

    save_weights(
    "checkpoints/model_saved.pkl",
    c1_w=c1_w,
    c1_b=c1_b,
    l1_w=l1_w,
    l1_b=l1_b)

    plot_metrics(epoch_losses, epoch_accs)

    print("Training complete.")
    print("Weights saved in ./model_saved.pkl")
    print("Plots saved in ./plots/")

def save_weights(path, **tensors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {name: t.data for name, t in tensors.items()},
            f
        )

def plot_metrics(losses, accs, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(losses) + 1))

    # Loss vs Epoch
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
    plt.close()

    # Accuracy vs Epoch
    plt.figure()
    plt.plot(epochs, accs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "acc_vs_epoch.png"))
    plt.close()

def print_model_stats(C_OUT, NUM_CLASSES):
    
    conv_params = C_OUT * 3 * 3 * 3 + C_OUT
    fc_params = NUM_CLASSES * (C_OUT * 16 * 16) + NUM_CLASSES
    total_params = conv_params + fc_params

    conv_macs = C_OUT * 32 * 32 * (3 * 3 * 3)
    fc_macs = NUM_CLASSES * (C_OUT * 16 * 16)
    total_macs = conv_macs + fc_macs

    total_flops = 2 * total_macs

    print("Model Statistics")
    print("----------------")
    print(f"Conv Params     : {conv_params}")
    print(f"FC Params       : {fc_params}")
    print(f"Total Params    : {total_params}")
    print()
    print(f"Conv MACs       : {conv_macs}")
    print(f"FC MACs         : {fc_macs}")
    print(f"Total MACs      : {total_macs}")
    print()
    print(f"Total FLOPs     : {total_flops}")


if __name__ == "__main__":
    train()


