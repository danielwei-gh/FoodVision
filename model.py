import torch
import torchvision

from torch import nn

def create_vit(num_classes: int, compile: bool = False):
    """Creates a pretrained ViT-B/16 feature extractor and its inference transforms.

    Args:
        num_classes (int): number of downstream classes.
        compile (bool): whether to speed up the model with `torch.compile`.

    Returns:
        model (torch.nn.Module): Pretrained ViT-B/16 feature extractor.
        model_transforms (torchvision.transforms): ViT-B/16 inference transforms.
    """
    # Create ViT-B/16 pretrained weights, transforms, and model
    model_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model_transforms = model_weights.transforms()
    model = torchvision.models.vit_b_16(weights=model_weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False
    
    # Change classifier head to zero-initialized linear layer for fine-tuning
    model.heads = nn.Linear(in_features=768, out_features=num_classes)
    nn.init.zeros_(model.heads.weight)
    nn.init.zeros_(model.heads.bias)

    if compile:
        model = torch.compile(model)

    return model, model_transforms
