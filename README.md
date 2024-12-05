# Depth Estimation

## Installation

1. Clone the repository:
  ```sh
  git clone https://github.com/fsudjatmiko/depth-estimation.git
  cd depth-estimation
  ```

2. Create a virtual environment and activate it:
  ```sh
  python3 -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```

3. Install the required dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

1. Download an image and place the URL in `main.py`:
  ```py
  url = "https://putyourphotohere.example/image.jpg"  # Replace this URL with your own image URL
  ```

2. Run the main script:
  ```sh
  python main.py
  ```

3. The results will be saved in `nearest_objects.json` and visualized.

## Project Structure

```
depth-estimation/
├── depth_estimation.py       # Depth estimation logic
├── object_detection.py       # Object detection logic
├── visualization.py          # Visualization logic
├── main.py                   # Main script to run the project
├── requirements.txt          # Project dependencies
├── nearest_objects.json      # Output file with detected objects
└── README.md                 # Project documentation
```
