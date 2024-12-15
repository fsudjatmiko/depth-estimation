# Depth Estimation and Object Detection

This project performs depth estimation and object detection on images using pre-trained models.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/fsudjatmiko/depth-estimation.git
    cd depth-estimation
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Update the image URL in `image_processor.py`:
    ```py
    url = "https://putyourphotohere.example/image.jpg"  # Replace this URL with your own image URL
    ```

2. Run the script:
    ```sh
    python image_processor.py
    ```

3. The results will be saved in `nearest_objects.json`.

## Project Structure
