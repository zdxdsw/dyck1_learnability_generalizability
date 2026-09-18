from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_dir = Path("./figures/")
png_paths = sorted(img_dir.glob("*.png"))
DPI = 300
X = "neg"
save_path = img_dir / f"{X}_length_distribution_figure.png"

suptitle = r"Length Distribution of Positive Sequences" if X == "pos" else r"Length Distribution of Negative Sequences"
# filter for X and train_/val_ prefix
png_paths = [p for p in png_paths if f"_{X}_" in p.name and not "test_" in p.name]

images = [Image.open(p).convert("RGB") for p in png_paths]

# Check image sizes
# sizes = [img.size for img in images]  # each is (width, height)
# if len(set(sizes)) != 1:
#     print("Images do not all have the same size:")
#     for p, size in zip(png_paths, sizes): print(p.name, size)
# else: print(f"All images have the same size: {sizes[0]}")

def name_to_title_and_col_row_idx(name):
    d = int(name.split("_d_")[1].split("_")[0])
    row_idx = d - 2
    if "train" in name:
        col_idx = 0
        title = rf"$\mathcal{{D}}^{{tr}}_{{d={d}}}$"
    else:
        col_idx = 1
        title = rf"$\mathcal{{D}}^{{pg}}_{{d={d}}}$ / $\mathcal{{D}}^{{dg}}_{{d={d}}}$"
    return title, col_idx, row_idx


def recolor_green_pixels(
    img,
    new_color, # RGB replacement color
    green_threshold=200, # Minimum green-channel value required to count as green.
    dominance_margin=70 # How much larger G must be than R and B.
):

    img = img.convert("RGB")
    arr = np.array(img)

    # Cast to avoid uint8 overflow
    arr_int = arr.astype(np.int16)

    r = arr_int[..., 0]
    g = arr_int[..., 1]
    b = arr_int[..., 2]

    green_mask = (
        (g > green_threshold) &
        (g > r + dominance_margin) &
        (g > b + dominance_margin)
    )

    arr[green_mask] = new_color
    return Image.fromarray(arr)

# ---------------------------------------------------------------------------
crop_top = 75  # pixels to remove from the top

cropped_images = []

for img in images:
    width, height = img.size
    # Crop box: (left, upper, right, lower)
    cropped = img.crop((0, crop_top, width, height))
    cropped_images.append(cropped)

fig, axes = plt.subplots(
    7, 2,
    figsize=(8, 18),
    constrained_layout=True
)

for img, path in zip(cropped_images, png_paths):
    title, col_idx, row_idx = name_to_title_and_col_row_idx(path.name)
    # if col_idx>0:
    #     img = recolor_green_pixels(img, new_color=(255, 245, 51))
    ax = axes[row_idx, col_idx]
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

fig.savefig(save_path, bbox_inches="tight", dpi=DPI)