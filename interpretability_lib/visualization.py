import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_saliency_map(image: np.array, heatmap: np.array) -> None:
    """
    Display a heatmap.

    Args:
        image (np.ndarray): Original image on which to overlay a heatmap.
        heatmap (np.ndarray): The heatmap to display.
    """
    plt.figure(figsize=(15, 15))

    # Display the cropped image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Cropped Image')

    # Display the saliency map
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap=plt.cm.hot)
    plt.axis('off')
    plt.title('Saliency Map')

    # Display the saliency map applied on top of the cropped image
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.5)
    plt.imshow(heatmap, cmap=plt.cm.hot, alpha=0.5)
    plt.axis('off')
    plt.title('Saliency Map on Image')

    plt.show()
