
import os
import torch
import gradio as gr

from timeit import default_timer
from model import create_vit

# Set up target device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get class names
with open('food_categories.txt', 'r') as f:
    CLASS_NAMES = [food_name.strip() for food_name in f.readlines()]

# Instantiate model and inference transforms
vit, vit_transforms = create_vit(num_classes=len(CLASS_NAMES))

# Load saved weights into model
vit.load_state_dict(torch.load(f='pretrained_vit_feature_extractor_food101.pth',
                               map_location=DEVICE))

# Move model to target device
vit.to(DEVICE)

# Create prediction function for Gradio app
def predict(img) -> tuple[dict, float]:
    """
    Prediction function for Gradio interface to wrap around. Returns prediction dictionary and inference time.
    """
    # Start timer
    start_time = default_timer()

    # Transform image and add batch dimension
    img = vit_transforms(img).unsqueeze(dim=0).to(DEVICE)

    # Set model to evaluate mode and turn on inference context manager
    vit.eval()
    with torch.inference_mode():
        probs = torch.softmax(vit(img), dim=1)
    
    # Create prediction label to prediction probability dictionary for each class (required format for Gradio's Label component)
    predictions = {CLASS_NAMES[i]: probs[0][i].item() for i in range(len(CLASS_NAMES))}

    # End timer and calculate prediction time
    end_time = default_timer()
    pred_time = round(end_time - start_time, 3)

    return predictions, pred_time

# Create title and description for Gradio app
TITLE = 'Food Image Classifier (ViT-B/16)'
DESCRIPTION = 'An food image classifier fine-tuned on the Food101 dataset, using a pretrained ViT-B/16 model from PyTorch.'

# Create example list for Gradio app
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Create Gradio Interface
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=5, label='Predictions'), gr.Number(label='Inference time (s)')],
                    examples=example_list,
                    title=TITLE,
                    description=DESCRIPTION)

# Launch app
demo.launch()
