import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def load_and_display_image(ax, image, title, border_color=None):
    """Helper function to display an image with an optional border."""
    if isinstance(image, str):
        # If image is a file path, load it
        img = mpimg.imread(image)
    else:
        # Assume the image is a NumPy array
        img = image

    ax.imshow(img)
    ax.set_title(title, fontsize=12, loc="center", pad=10)
    ax.axis("off")  # Hide axes

    if border_color:
        rect = patches.Rectangle(
            (0, 0),
            img.shape[1],
            img.shape[0],
            linewidth=10,
            edgecolor=border_color,
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_list_of_images(source_image, nearest_neighbors_paths):
    # If source image is None, plot a white image
    if source_image is None:
        source_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

    # Combine the source image and nearest neighbor images into a single list
    all_images = [source_image] + nearest_neighbors_paths

    # Create a figure with a 1x5 grid (one row with five images)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5), constrained_layout=True)

    # Load and display the source image with a blue border
    load_and_display_image(axes[0], all_images[0], "Query Image", border_color="blue")

    # Load and display the second image with a green border
    load_and_display_image(axes[1], all_images[1], "Neighbor 1", border_color="green")

    # Load and display the remaining images without borders
    for i, image in enumerate(all_images[2:], start=2):
        load_and_display_image(axes[i], image, f"Neighbor {i-1}")

    plt.show()
