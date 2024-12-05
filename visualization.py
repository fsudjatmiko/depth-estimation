import matplotlib.pyplot as plt
import numpy as np

def visualize_depth_map(image, depth_in_meters, detected_objects):
    # Visualize the depth map with object detection boxes
    fig, ax = plt.subplots(1, figsize=(10, 6))  # Adjust the figure size as needed
    im = ax.imshow(depth_in_meters, cmap='plasma')  # Visualize the depth map

    for obj in detected_objects:
        x0, y0, x1, y1 = map(int, obj['bbox'])
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(depth_in_meters.shape[1], x1)
        y1 = min(depth_in_meters.shape[0], y1)
        avg_depth = np.nanmean(depth_in_meters[y0:y1, x0:x1])
        if not np.isnan(avg_depth) and avg_depth > 0:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='red', linewidth=3))
            text = f"{obj['label']}: {avg_depth:.2f} meters"
            ax.text(x0, y0, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.colorbar(im, label='Depth (meters)')
    plt.axis('off')
    plt.show()
