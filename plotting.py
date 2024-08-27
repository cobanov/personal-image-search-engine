import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


def plot_list_of_images(source_image, nearest_neighbors_paths):

    # Combine the source image and nearest neighbor images into a single list
    all_images = [source_image] + nearest_neighbors_paths

    # Create a figure with a 1x5 grid (one row with five images)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5), constrained_layout=True)

    # Load and display the source image with a blue border
    img = mpimg.imread(all_images[0])
    axes[0].imshow(img)
    axes[0].set_title("Query Image", fontsize=12, loc="center", pad=10)
    axes[0].axis("off")  # Hide axes
    # Add blue border
    rect = patches.Rectangle(
        (0, 0),
        img.shape[1],
        img.shape[0],
        linewidth=10,
        edgecolor="blue",
        facecolor="none",
    )
    axes[0].add_patch(rect)

    # Load and display the second image with a green border
    img = mpimg.imread(all_images[1])
    axes[1].imshow(img)
    axes[1].set_title("Neighbor 1", fontsize=12, loc="center", pad=10)
    axes[1].axis("off")  # Hide axes
    # Add green border
    rect = patches.Rectangle(
        (0, 0),
        img.shape[1],
        img.shape[0],
        linewidth=10,
        edgecolor="green",
        facecolor="none",
    )
    axes[1].add_patch(rect)

    # Load and display the remaining images without borders
    for i, image_path in enumerate(all_images[2:], start=2):
        img = mpimg.imread(image_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Neighbor {i-1}", fontsize=12, loc="center", pad=10)
        axes[i].axis("off")  # Hide axes

    plt.show()
