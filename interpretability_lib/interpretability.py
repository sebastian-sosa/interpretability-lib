import torch


def compute_saliency_map(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute the saliency map of a given model for a given input.

    Args:
        model (torch.nn.Module): The model to compute the saliency map for.
        inputs (torch.Tensor): The inputs to the model.

    Returns:
        torch.Tensor: The computed saliency map.
    """
    model.eval()
    inputs.requires_grad_()
    outputs = model(inputs)
    outputs.backward(torch.ones_like(outputs))
    saliency_map = inputs.grad.data
    return saliency_map
