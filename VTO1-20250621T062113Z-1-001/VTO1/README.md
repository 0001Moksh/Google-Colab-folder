
# VTO 1: Virtual Try-On Application (POC)

This project, "VTO 1," is a Proof of Concept (POC) for a Virtual Try-On (VTO) application. It utilizes a pre-trained deep learning model (MobileNetV2) to combine a user's image with a product image, creating a visual representation of how the product might look on the user. The project also outlines the setup for a FastAPI-based web service to enable this functionality through an API.

-----

## Features

  * **Image Loading and Preprocessing:** Handles uploading user and product images from Google Colab, converting them to a usable format for OpenCV and TensorFlow.
  * **Feature Extraction (Conceptual):** Uses a pre-trained `MobileNetV2` model (without the top classification layer) to extract features from both the user and product images. *Note: In this POC, the extracted features `user_features` and `product_mask` are not directly used for a sophisticated "try-on" logic but rather for a simple image blending.*
  * **Basic Image Blending:** Implements a `try_on_model` function that resizes images and performs a simple weighted addition of the user and product images to create the "try-on" effect.
  * **Output Saving:** Saves the generated output image to a file.
  * **FastAPI Integration (Planned/Incomplete):** Includes code snippets for setting up a FastAPI application to expose the try-on functionality as a web API, allowing users to upload images and receive the blended output.

-----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * Google Colab (recommended for running the notebook code and handling file uploads)

### Libraries

You can install the necessary Python libraries using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib pillow fastapi uvicorn
```

-----

## Usage

The project is primarily designed to be run in a Google Colab environment.

### 1\. Run the Initial Try-On Script (Colab Notebook)

The first part of the code demonstrates the core try-on functionality:

1.  **Install Dependencies:**
    ```python
    !pip install tensorflow opencv-python numpy matplotlib pillow
    ```
2.  **Import Libraries:**
    ```python
    import tensorflow as tf
    import cv2
    import numpy as np
    from google.colab import files
    from PIL import Image
    import io
    import os
    ```
3.  **Load Pre-trained Model:**
    ```python
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    ```
4.  **Upload Images:** The script will prompt you to upload a "USER image" and a "PRODUCT image" using the Colab file upload widget.
    ```python
    print("Upload USER image")
    user_file = files.upload()
    print("Upload PRODUCT image")
    product_file = files.upload()
    ```
5.  **Process and Generate Output:** The `try_on_model` function will resize the images, perform a weighted blend, and save the result as `output_image.jpg`.
    ```python
    output_image = try_on_model(user_image, product_image)
    if output_image is not None:
        output_path = "output_image.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"Output saved as {output_path}")
    else:
        print("Failed to generate output image.")
    ```

### 2\. FastAPI Web API (Planned Usage)

The latter part of the code sets up a FastAPI application. To run this as a web service:

1.  **Install FastAPI and Uvicorn:**

    ```bash
    !pip install fastapi uvicorn
    ```

2.  **Define FastAPI Application:** The provided code snippet defines a basic FastAPI app with a root endpoint and a `/try-on` endpoint that accepts image uploads.
    *Note: The `UploadFile` import is missing, and the `try_on_model` function should be accessible to this part of the code.*

3.  **Run the FastAPI Server:** (This would typically be run in a separate terminal or Colab cell after saving the FastAPI code to a file named `app.py`)

    ```bash
    !uvicorn app:app --reload
    ```

    Once running, you would access the API typically at `http://127.0.0.1:8000` (or the appropriate Colab proxy URL). You could then make POST requests to `/try-on` with multipart form data containing the user and product images.

-----

## Limitations and Future Improvements

  * **Simple Blending:** The current `try_on_model` uses a basic weighted average, which is not a true "try-on" mechanism. A more advanced VTO system would involve:
      * **Body Part Segmentation:** Identifying and isolating the clothing region on the product image.
      * **Human Pose Estimation:** Detecting the pose of the person in the user image.
      * **Image Warping and Blending:** Warping the product image to fit the user's body shape and pose more realistically.
      * **Generative Models:** Using techniques like GANs (Generative Adversarial Networks) or diffusion models for highly realistic try-on results.
  * **Model Usage:** The `MobileNetV2` feature extraction is present but its outputs (`user_features`, `product_mask`) are not effectively utilized for a sophisticated try-on. These features would typically be fed into a more complex VTO network.
  * **Error Handling:** Enhance error handling, especially for image processing failures and API requests.
  * **Scalability:** For production use, consider optimizing image processing, GPU utilization, and API serving.

-----
