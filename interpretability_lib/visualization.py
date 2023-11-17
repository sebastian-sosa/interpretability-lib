from typing import List

import matplotlib.pyplot as plt
import numpy as np


class VisualizationModule:
    def display_heatmap(self, heatmap: np.ndarray) -> None:
        """
        Display a heatmap.

        Args:
            heatmap (np.ndarray): The heatmap to display.
        """
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

    def display_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> None:
        """
        Display a heatmap overlayed on an image.

        Args:
            image (np.ndarray): The image to display.
            heatmap (np.ndarray): The heatmap to overlay.
        """
        plt.imshow(image)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest', alpha=0.5)
        plt.show()

    def display_image(self, image: np.ndarray) -> None:
        """
        Display an image.

        Args:
            image (np.ndarray): The image to display.
        """
        plt.imshow(image)
        plt.show()

    def display_grid(self, images: List[np.ndarray], titles: List[str]) -> None:
        """
        Display a grid of images.

        Args:
            images (List[np.ndarray]): The images to display.
            titles (List[str]): The titles for the images.
        """
        n = len(images)
        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
        for i, (img, title) in enumerate(zip(images, titles)):
            axs[i].imshow(img)
            axs[i].set_title(title)
            axs[i].axis('off')
        plt.show()
