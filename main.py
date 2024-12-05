import requests
from PIL import Image
import json
from depth_estimation import estimate_depth
from object_detection import detect_objects
from visualization import visualize_depth_map

# Load an image
url = "https://putyourphotohere.example/image.jpg"  # Replace this URL with your own image URL
image = Image.open(requests.get(url, stream=True).raw)

# Estimate depth
depth_in_meters = estimate_depth(image)

# Perform object detection
detected_objects = detect_objects(image, depth_in_meters)

# Save the results to a JSON file
with open("nearest_objects.json", "w") as f:
    json.dump(detected_objects, f, indent=4)

# Print the results
print(json.dumps(detected_objects, indent=4))

# Visualize the depth map with object detection boxes
visualize_depth_map(image, depth_in_meters, detected_objects)
