import torch
from torch.autograd import Variable


def compute_saliency_map(input_image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Compute the saliency map of a given model for a given input.

    Args:
        input_image (torch.Tensor): The input image.
        model (torch.nn.Module): The model to compute the saliency map for.

    Returns:
        torch.Tensor: The computed saliency map.
    """
    input_image = Variable(input_image, requires_grad=True)
    output = model(input_image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency_map = input_image.grad.data
    saliency_map = saliency_map.abs()
    saliency_map, _ = torch.max(saliency_map, dim=1)
    saliency_map = saliency_map.squeeze()
    return saliency_map
