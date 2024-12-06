import cv2
import numpy as np
import json
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a YOLOv8 model file

# Load stereo camera parameters
fs = cv2.FileStorage("stereo_params.yml", cv2.FILE_STORAGE_READ)
Q = fs.getNode("Q").mat()  # Disparity-to-depth mapping matrix
fs.release()

# Stereo camera video capture
cap_left = cv2.VideoCapture(0)  # Left camera
cap_right = cv2.VideoCapture(1)  # Right camera

# Stereo block matcher
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        print("Error capturing frames")
        break

    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(gray_left, gray_right)
    disparity = np.float32(disparity) / 16.0

    # Compute depth map
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Detect objects using YOLOv8
    results = model(frame_left, stream=False)  # Perform detection on left frame

    objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = map(int, box.xyxy[0])
            label = result.names[cls]

            # Calculate object depth
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth = depth_map[center_y, center_x][2]  # Get Z-coordinate

            if np.isfinite(depth) and depth > 0:  # Check for valid depth
                objects.append({
                    "label": label,
                    "confidence": float(conf),
                    "depth_meters": float(depth),
                    "bounding_box": [x1, y1, x2, y2]
                })

    # Sort objects by depth and take the four nearest
    objects = sorted(objects, key=lambda obj: obj["depth_meters"])[:4]

    # Convert to JSON format
    json_output = json.dumps({"detected_objects": objects}, indent=4)
    print(json_output)

    # Display frames with bounding boxes and depth
    for obj in objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        depth = obj["depth_meters"]
        label = obj["label"]

        # Draw bounding box and label
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_left, f"{label} {depth:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Left Frame", frame_left)
    cv2.imshow("Disparity", disparity / disparity.max())

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
