import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

kirby_bw = np.array([
    [0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0],
    [0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0],
    [1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    [0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [0,1,0,0,1,1,0,0,0,1,1,0,0,0,1,0],
    [1,0,0,0,0,0,1,1,1,0,1,0,0,1,0,0],
    [0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])

# 1. Binary image (black and white)
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(kirby_bw, cmap='gray_r', interpolation='nearest')
ax.axis('off')
plt.savefig('binary-image.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 2. Grayscale image (with gradients)
kirby_gray = kirby_bw.astype(float)
kirby_gray[kirby_bw == 0] = np.random.uniform(0.7, 0.9, size=(kirby_bw == 0).sum())
kirby_gray[kirby_bw == 1] = np.random.uniform(0.1, 0.3, size=(kirby_bw == 1).sum())
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(kirby_gray, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
ax.axis('off')
plt.savefig('grayscale-image.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 3. Color image (pink Kirby!)
kirby_color = np.zeros((16, 16, 3))
kirby_color[kirby_bw == 0] = [1.0, 0.7, 0.8]  # Pink body
kirby_color[kirby_bw == 1] = [0.2, 0.2, 0.2]  # Dark outline
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(kirby_color, interpolation='nearest')
ax.axis('off')
plt.savefig('color-image.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 4. Pixel grid with numbers (showing a small portion)
fig, ax = plt.subplots(figsize=(6, 4))
small_region = kirby_bw[2:7, 2:8]
ax.imshow(small_region, cmap='gray_r', interpolation='nearest')
for i in range(small_region.shape[0]):
    for j in range(small_region.shape[1]):
        color = 'white' if small_region[i, j] == 1 else 'black'
        ax.text(j, i, str(small_region[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Each pixel has a value (0 or 1)', fontsize=12)
plt.savefig('pixel-grid.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

print("Generated: binary-image.png, grayscale-image.png, color-image.png, pixel-grid.png")
