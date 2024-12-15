import requests
from PIL import Image
import json
from depth_estimation import estimate_depth
from object_detection import detect_objects

def process_image(image):
    # Estimate depth
    depth_in_meters = estimate_depth(image)

    # Perform object detection
    detected_objects = detect_objects(image, depth_in_meters)

    # Save the results to a JSON file
    with open("nearest_objects.json", "w") as f:
        json.dump(detected_objects, f, indent=4)

    return detected_objects

if __name__ == "__main__":
    # Load an image from a URL
    url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSevT-0iO36ohAaLwusmtqf5t6hPtuAgCJ1KQ&s"  # Replace this URL with your own image URL
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Process the image
    process_image(image)
