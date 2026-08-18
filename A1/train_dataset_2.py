import custom_core_dataset_2 as core
import os, time, math, random, pickle

random.seed(42)

def print_model_stats(C_OUT, NUM_CLASSES):
    conv_params = C_OUT * 3 * 3 * 3 + C_OUT
    fc_params = NUM_CLASSES * (C_OUT * 16 * 16) + NUM_CLASSES
    total_params = conv_params + fc_params

    conv_macs = C_OUT * 32 * 32 * (3 * 3 * 3)
    fc_macs = NUM_CLASSES * (C_OUT * 16 * 16)
    total_macs = conv_macs + fc_macs

    print(f"Params: {total_params}, MACs: {total_macs}, FLOPs: {2*total_macs}")

def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())

def train():
    SOURCE_PATH = os.path.abspath(input("Give the path to dataset: "))
    print('Loading Data...')
    start_time = time.time()
    imgs, labels = core.load_dataset_2(SOURCE_PATH)
    end_time = time.time()
    print("Data Loading Time: ", end_time-start_time)

    train_data = list(zip(imgs, labels))
    random.shuffle(train_data)

    NUM_CLASSES = 100
    C_OUT = 16
    LR = 0.005
    EPOCHS = 10
    MOM = 0.9
    WD = 1e-4

    print_model_stats(C_OUT, NUM_CLASSES)

    c1_w = core.Tensor([C_OUT, 3, 3, 3])
    c1_b = core.Tensor([C_OUT])
    l1_w = core.Tensor([NUM_CLASSES, C_OUT * 16 * 16])
    l1_b = core.Tensor([NUM_CLASSES])

    for p in [c1_w, l1_w]:
        fan_in = p.shape[1] * p.shape[2] * p.shape[3] if len(p.shape) == 4 else p.shape[1]
        std = math.sqrt(2.0 / fan_in)
        p.data = [random.gauss(0, std) for _ in p.data]

    c1_b.data = [0.0] * len(c1_b.data)
    l1_b.data = [0.0] * len(l1_b.data)

    act_conv = core.Tensor([1, C_OUT, 32, 32])
    act_pool = core.Tensor([1, C_OUT, 16, 16])
    pool_mask = core.Tensor([1, C_OUT, 32, 32])
    act_flat = core.Tensor([1, C_OUT * 16 * 16])
    act_logits = core.Tensor([1, NUM_CLASSES])
    grad_conv = core.Tensor([1, C_OUT, 32, 32])
    grad_pool = core.Tensor([1, C_OUT, 16, 16])

    epoch_losses = []
    epoch_accs = []

    for ep in range(EPOCHS):
        correct = 0
        loss_sum = 0

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

        print(f"Epoch {ep+1} | Loss {avg_loss:.4f} | Acc {acc:.2%}")
        LR *= 0.9

    save_pickle("checkpoints/model_dataset_2.pkl", {
        "c1_w": list(c1_w.data),
        "c1_b": list(c1_b.data),
        "l1_w": list(l1_w.data),
        "l1_b": list(l1_b.data),
    })

    save_pickle("checkpoints/train_logs_dataset_2.pkl", {
        "losses": epoch_losses,
        "accs": epoch_accs
    })

    print("Training finished. Weights & logs saved.")
    os._exit(0)

if __name__ == "__main__":
    train()

