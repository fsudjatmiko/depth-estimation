from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
detection_model = YOLO('yolov8n.pt')

def detect_objects(image, depth_in_meters):
    # Perform object detection using YOLOv8
    results = detection_model(image)

    # Set a confidence threshold
    confidence_threshold = 0.5

    # Collect detected objects with their average depth
    detected_objects = []

    for result in results:
        for box in result.boxes:
            if box.conf >= confidence_threshold:
                x0, y0, x1, y1 = map(int, box.xyxy[0])
                # Ensure the bounding box coordinates are within the valid range
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(depth_in_meters.shape[1], x1)
                y1 = min(depth_in_meters.shape[0], y1)
                avg_depth = np.nanmean(depth_in_meters[y0:y1, x0:x1])  # Use nanmean to ignore NaN values
                if not np.isnan(avg_depth) and avg_depth > 0:  # Ignore objects with 0 meters depth
                    detected_objects.append({
                        "label": detection_model.names[int(box.cls)],  # Convert to native int type and get label name
                        "depth_meters": float(avg_depth),  # Convert to native float type and include unit
                        "bbox": [x0, y0, x1, y1]  # Include bounding box coordinates
                    })

    # Sort the objects by their average depth and select the 4 nearest objects
    detected_objects = sorted(detected_objects, key=lambda x: x["depth_meters"])[:4]

    return detected_objects
