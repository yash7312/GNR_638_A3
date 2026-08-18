import custom_core_dataset_2 as core
import os, time, pickle

def load_weights(path, **tensors):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    for name, t in tensors.items():
        t.data = saved[name]

def evaluate():
    SOURCE_PATH = os.path.abspath(input("Give the path to dataset: "))
    MODEL_PATH = input("Give the path to saved model (.pkl): ")

    print(f"Dataset path: {SOURCE_PATH}")
    print(f"Model path:   {MODEL_PATH}")

    print("Loading Dataset-2...")
    t0 = time.time()
    imgs, labels = core.load_dataset_2(SOURCE_PATH)
    print(f"Loaded {len(imgs)} samples in {time.time() - t0:.2f}s")

    NUM_CLASSES = 100
    C_OUT = 16

    c1_w = core.Tensor([C_OUT, 3, 3, 3])
    c1_b = core.Tensor([C_OUT])
    l1_w = core.Tensor([NUM_CLASSES, C_OUT * 16 * 16])
    l1_b = core.Tensor([NUM_CLASSES])

    load_weights(
        MODEL_PATH,
        c1_w=c1_w,
        c1_b=c1_b,
        l1_w=l1_w,
        l1_b=l1_b
    )

    act_conv = core.Tensor([1, C_OUT, 32, 32])
    act_pool = core.Tensor([1, C_OUT, 16, 16])
    pool_mask = core.Tensor([1, C_OUT, 32, 32])
    act_flat = core.Tensor([1, C_OUT * 16 * 16])
    act_logits = core.Tensor([1, NUM_CLASSES])

    correct = 0
    total = len(imgs)

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    print("Evaluating...")
    t1 = time.time()

    for img, y in zip(imgs, labels):
        # No gradients needed
        for t in [act_conv, act_pool, act_flat, act_logits, img]:
            t.zero_grad()

        core.conv2d_fwd(img, c1_w, c1_b, act_conv, 1, 1)
        core.relu_fwd(act_conv)
        core.maxpool_fwd(act_conv, act_pool, pool_mask, 2, 2)
        core.flatten(act_pool, act_flat)
        core.linear_fwd(act_flat, l1_w, l1_b, act_logits)

        pred = act_logits.data.index(max(act_logits.data))

        if pred == y:
            correct += 1
            class_correct[y] += 1
        class_total[y] += 1

    acc = correct / total

    print("\nEvaluation Results")
    print("------------------")
    print(f"Overall Accuracy: {acc:.2%}")
    print(f"Evaluation Time:  {time.time() - t1:.2f}s")

    print("\nPer-class accuracy (first 20 classes):")
    for i in range(20):
        if class_total[i] > 0:
            print(f"Class {i:02d}: {class_correct[i] / class_total[i]:.2%}")
        else:
            print(f"Class {i:02d}: N/A")

if __name__ == "__main__":
    evaluate()

