
# Rice Grain Measurement Application

This repository contains a Streamlit-based web application for analyzing and measuring rice grains from uploaded images. The application processes the image to detect and classify rice grains based on their dimensions.

## Features

- Upload images in various formats (JPG, JPEG, PNG, TIFF).
- Automatically process the image to detect rice grains.
- Fit ellipses around detected rice grains and classify them into three categories:
  - Whole grains (Green)
  - Broken grains (Red)
  - Normal grains (Orange)
- Display the original and processed images side by side.
- Provide detailed measurements of each rice grain in millimeters.
- Display counts for each category of rice grains.

## Getting Started

### Prerequisites

To run this application, you need to have the following installed on your system:

- Python 3.6 or higher
- Streamlit
- OpenCV
- NumPy

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/rice-grain-measurement.git
    cd rice-grain-measurement
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the application, use the following command:

```bash
streamlit run app.py
```

This will start a local Streamlit server, and you can access the application in your web browser at `http://localhost:8501`.

## Usage

1. Upload an image of rice grains using the file uploader.
2. The application will process the image and display the original and processed images side by side.
3. The processed image will show ellipses around detected rice grains, color-coded based on their classification.
4. Detailed measurements of each rice grain will be displayed below the images.
5. Counts of whole, broken, and normal grains will also be displayed.

## Code Overview

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of required Python packages.

### Functions

- `process_image(image)`: Function to process the uploaded image, detect rice grains, and classify them.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
