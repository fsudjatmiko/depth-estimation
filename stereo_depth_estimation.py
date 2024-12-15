import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import numpy as np
from PIL import Image
import json
from object_detection import detect_objects
import requests

depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
depth_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-base-384")

def estimate_depth(left_image, right_image):
    depth_inputs_left = depth_image_processor(images=left_image, return_tensors="pt")
    depth_inputs_right = depth_image_processor(images=right_image, return_tensors="pt")

    with torch.no_grad():
        depth_outputs_left = depth_model(**depth_inputs_left)
        depth_outputs_right = depth_model(**depth_inputs_right)
        predicted_depth_left = depth_outputs_left.predicted_depth
        predicted_depth_right = depth_outputs_right.predicted_depth

    scaling_factor = 0.001 
    depth_in_meters_left = predicted_depth_left * scaling_factor
    depth_in_meters_right = predicted_depth_right * scaling_factor

    depth_in_meters_left = depth_in_meters_left.squeeze().cpu().numpy()
    depth_in_meters_right = depth_in_meters_right.squeeze().cpu().numpy()

    depth_in_meters_left = np.max(depth_in_meters_left) - depth_in_meters_left
    depth_in_meters_right = np.max(depth_in_meters_right) - depth_in_meters_right

    depth_in_meters_combined = (depth_in_meters_left + depth_in_meters_right) / 2

    depth_in_meters_resized = np.array(Image.fromarray(depth_in_meters_combined).resize(left_image.size, Image.BILINEAR))

    return depth_in_meters_resized

def process_images(left_image, right_image):

    depth_in_meters = estimate_depth(left_image, right_image)
    detected_objects = detect_objects(left_image, depth_in_meters)

    with open("stereo_nearest_objects.json", "w") as f:
        json.dump(detected_objects, f, indent=4)

    return detected_objects

if __name__ == "__main__":
    left_url = "https://i.ibb.co.com/pn1CChh/kiri.jpg"  
    right_url = "https://i.ibb.co.com/371fLwN/kanan.jpg"  
    left_image = Image.open(requests.get(left_url, stream=True).raw)
    right_image = Image.open(requests.get(right_url, stream=True).raw)
    
    process_images(left_image, right_image)
