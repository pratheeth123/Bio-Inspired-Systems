import cv2
import numpy as np
import random
import urllib.request
import matplotlib.pyplot as plt

def url_to_image(url, flag=cv2.IMREAD_GRAYSCALE):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, flag)
    return img

url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
img = url_to_image(url)
img = cv2.resize(img, (128, 128))
h, w = img.shape

def fitness(thresh):
    fg = img[img >= thresh]
    bg = img[img < thresh]
    if len(fg) == 0 or len(bg) == 0:
        return 0
    mean_fg = np.mean(fg)
    mean_bg = np.mean(bg)
    w_fg = len(fg)/img.size
    w_bg = len(bg)/img.size
    return w_fg*w_bg*(mean_fg - mean_bg)**2

POP_SIZE = 20
GENS = 30
MUT_RATE = 0.2

population = [random.randint(0, 255) for _ in range(POP_SIZE)]
best_thresh = None
best_score = 0

for gen in range(GENS):
    scores = [fitness(t) for t in population]
    idx = np.argmax(scores)
    if scores[idx] > best_score:
        best_score = scores[idx]
        best_thresh = population[idx]
    print(f"Generation {gen+1}: Best Threshold = {best_thresh}, Score = {best_score:.2f}")
    
    selected_idx = np.argsort(scores)[::-1][:POP_SIZE//2]
    selected = [population[i] for i in selected_idx]
    next_gen = []
    while len(next_gen) < POP_SIZE:
        parent = random.choice(selected)
        child = parent
        if random.random() < MUT_RATE:
            child = min(255, max(0, parent + random.randint(-10,10)))
        next_gen.append(child)
    population = next_gen

_, segmented = cv2.threshold(img, best_thresh, 255, cv2.THRESH_BINARY)

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title(f"Segmented (Thresh={best_thresh})")
plt.imshow(segmented, cmap="gray")
plt.axis("off")
plt.show()
