import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import clip
from PIL import Image

# ======================
# CONFIG
# ======================
PATCH_DIR = "patches"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.5

# ======================
# LOAD CLIP (LOCAL)
# ======================
print("Loading CLIP...")
model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE,
    download_root="models"
)
model.eval()

# ======================
# LOAD PATCHES
# ======================
def load_patches(folder):
    patches = {}
    for f in os.listdir(folder):
        if f.endswith(".png"):
            idx = int(f.split("_")[1].split(".")[0])
            patches[idx] = cv2.imread(os.path.join(folder, f))
    return patches

# ======================
# ROTATIONS
# ======================
def rotations(img):
    return [
        img,
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

# ======================
# EDGE SCORE
# ======================
def edge_score(a, b, direction):
    k = 30 # hardcoded?
    if direction == "right":
        return -np.mean((a[:, -k:] - b[:, :k])**2)
    else:
        return -np.mean((a[-k:, :] - b[:k, :])**2)

# ======================
# BUILD GRID
# ======================
def build_grid(patches):
    keys = list(patches.keys())
    size = int(np.sqrt(len(keys)))

    grid = [[None]*size for _ in range(size)]
    used = set()

    grid[0][0] = (0, patches[0])
    used.add(0)

    print("Stitching image...")

    for i in tqdm(range(size)):
        for j in range(size):
            if i == 0 and j == 0:
                continue

            best_score = -1e18
            best_choice = None

            for idx in keys:
                if idx in used:
                    continue

                for rot in rotations(patches[idx]):
                    score = 0

                    if j > 0:
                        score += edge_score(grid[i][j-1][1], rot, "right")

                    if i > 0:
                        score += edge_score(grid[i-1][j][1], rot, "bottom")

                    if score > best_score:
                        best_score = score
                        best_choice = (idx, rot)

            if best_choice is None:
                for idx in keys:
                    if idx not in used:
                        best_choice = (idx, patches[idx])
                        break

            grid[i][j] = best_choice
            used.add(best_choice[0])

    return grid

# ======================
# STITCH IMAGE
# ======================
def stitch(grid):
    rows = []
    for row in grid:
        rows.append(np.hstack([cell[1] for cell in row]))
    return np.vstack(rows)

# ======================
# INFORMATIVE PATCHES (KEY FIX)
# ======================
def get_informative_patches(img, size=224, stride=64):
    h, w, _ = img.shape
    patches = []

    for y in range(0, h-size, stride):
        for x in range(0, w-size, stride):

            patch = img[y:y+size, x:x+size]

            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            edges = np.mean(cv2.Canny(gray, 100, 200))
            variance = np.var(gray)

            score = edges + variance

            if score > 20:   # keep only informative hardcoded
                patches.append(patch)

    return patches

# ======================
# EXTRA ZOOM PATCHES
# ======================
def get_zoom_patches(img):
    h, w, _ = img.shape
    crops = []

    crops.append(img[h//4:3*h//4, w//4:3*w//4])
    crops.append(img[:h//2, :w//2])
    crops.append(img[:h//2, w//2:])
    crops.append(img[h//2:, :w//2])
    crops.append(img[h//2:, w//2:])

    return crops

# ======================
# CLIP SCORE
# ======================
def clip_score(image, texts):

    image = preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    text = clip.tokenize(texts).to(DEVICE)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    return probs

# ======================
# ANSWER FUNCTION
# ======================
def answer_question(question, options, full_map):

    texts = [question + ". " + opt for opt in options]

    best_score = 0
    best_option = 5

    patches = get_informative_patches(full_map)
    patches += get_zoom_patches(full_map)

    for patch in patches:
        probs = clip_score(patch, texts)

        idx = np.argmax(probs)
        score = probs[idx]

        if score > best_score:
            best_score = score
            best_option = idx + 1

    # full image also
    probs = clip_score(full_map, texts)
    idx = np.argmax(probs)
    score = probs[idx]

    if score > best_score:
        best_score = score
        best_option = idx + 1

    if best_score < CONF_THRESHOLD:
        return 5

    return best_option

# ======================
# MAIN
# ======================
def main():
    patches = load_patches(PATCH_DIR)

    grid = build_grid(patches)
    full_map = stitch(grid)

    cv2.imwrite("reconstructed_map.png", full_map)

    df = pd.read_csv(TEST_CSV)
    option_cols = [c for c in df.columns if "option" in c.lower()]

    answers = []

    print("Answering questions...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        qid = row["id"]
        question = row["question"]
        options = [row[c] for c in option_cols[:4]]

        ans = answer_question(question, options, full_map)

        answers.append({
            "id": qid,
            "question_num": qid,
            "option": ans
        })

    pd.DataFrame(answers).to_csv(OUTPUT_CSV, index=False)

    print("Done! Submission saved.")

# ======================
# RUN
# ======================
if __name__ == "__main__":
    main()