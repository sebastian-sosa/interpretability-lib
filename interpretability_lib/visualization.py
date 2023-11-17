from typing import List

import matplotlib.pyplot as plt
import numpy as np


class VisualizationModule:
    def display_heatmap(self, heatmap: np.ndarray) -> None:
        """
        Display a heatmap.

        Args:
            heatmap (np.ndarray): The heatmap to display. Expected shape is (height, width).
        """
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

    def display_overlay(self, image: np.ndarray, heatmap: np.ndarray, agg_func=np.amax) -> None:
        """
        Display a heatmap overlayed on an image.

        Args:
            image (np.ndarray): The image to display. Expected shape is (height, width, channels).
            heatmap (np.ndarray): The heatmap to overlay. 
                Expected shape is (height, width, channels) or (height, width).
            agg_func (function, optional): The aggregation function to apply across the channel
                dimension of the heatmap if it has 3 dimensions. Defaults to np.amax.
        """
        # If the heatmap has 3 dimensions, aggregate across the channel dimension
        # if len(heatmap.shape) == 3:
        #     heatmap = agg_func(heatmap, axis=0)
        if len(heatmap.shape) == 2:
            heatmap = heatmap[None, :, :]  # Add a singleton dimension

        plt.imshow(image)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest', alpha=0.5)
        plt.show()

    def display_image(self, image: np.ndarray) -> None:
        """
        Display an image.

        Args:
            image (np.ndarray): The image to display. Expected shape is (height, width, channels).
        """
        plt.imshow(image)
        plt.show()

    def display_grid(self, images: List[np.ndarray], titles: List[str]) -> None:
        """
        Display a grid of images.

        Args:
            images (List[np.ndarray]): The images to display. 
                Each image should have shape (height, width, channels).
            titles (List[str]): The titles for the images.
        """
        n = len(images)
        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
        for i, (img, title) in enumerate(zip(images, titles)):
            axs[i].imshow(img)
            axs[i].set_title(title)
            axs[i].axis('off')
        plt.show()
