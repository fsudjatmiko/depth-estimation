import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import numpy as np
from PIL import Image

# Load the depth estimation model and feature extractor
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

def estimate_depth(image):
    # Preprocess the image for depth estimation
    depth_inputs = depth_feature_extractor(images=image, return_tensors="pt")

    # Perform depth estimation
    with torch.no_grad():
        depth_outputs = depth_model(**depth_inputs)
        predicted_depth = depth_outputs.predicted_depth

    # Convert depth map to meters
    scaling_factor = 0.1  # Adjust this value as needed
    depth_in_meters = predicted_depth * scaling_factor

    # Convert the tensor to a numpy array for further processing
    depth_in_meters = depth_in_meters.squeeze().cpu().numpy()

    # Ensure the depth map is correctly oriented
    depth_in_meters = np.max(depth_in_meters) - depth_in_meters

    # Resize the depth map to match the original image dimensions
    depth_in_meters_resized = np.array(Image.fromarray(depth_in_meters).resize(image.size, Image.BILINEAR))

    return depth_in_meters_resized
