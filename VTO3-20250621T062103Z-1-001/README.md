
# VTO 3: Virtual Try-On with Pre-processing and Neural Networks

This project, "VTO 3," builds upon previous iterations by integrating essential pre-processing steps – background removal and person segmentation – to create a more robust (though still simplified) Virtual Try-On (VTO) pipeline. It continues to leverage PyTorch for defining and training neural networks responsible for warping the garment and synthesizing the final try-on image.

-----

## Features

  * **Comprehensive Pre-processing:**
      * **Background Removal:** Utilizes the `rembg` library to automatically remove backgrounds from both person and clothing images, ensuring cleaner input for the VTO model.
      * **Person Segmentation:** Integrates `Detectron2` (specifically Mask R-CNN) to segment the person from their background, which is crucial for more accurate garment placement. (Note: While segmentation is implemented, its direct use in the `GMM` and `TryOnSynthesizer` is simplified in this POC).
  * **PyTorch-based VTO Core:**
      * **`GMM` (Garment Motion Model / Warping Proxy):** A neural network module designed to learn a transformation that "warps" the clothing image to fit the person. In this simplified version, it acts as a conceptual warping network.
      * **`TryOnSynthesizer`:** Another neural network that takes the person's image and the (conceptually) warped garment to generate the final try-on composite.
  * **Training Loop:** Demonstrates a basic training pipeline including image loading, pre-processing, forward pass through the networks, L1 loss calculation, backpropagation, and optimization.
  * **Google Drive Integration:** Facilitates easy loading of input images from Google Drive within a Colab environment.
  * **Model Saving:** Saves the trained model's state dictionaries for future inference.
  * **Inference Example:** Shows how to load the trained model and run it on new test images.

-----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * Google Colab (highly recommended due to GPU requirements for PyTorch and `Detectron2`, and ease of environment setup).

### Libraries

You can install all necessary Python libraries and clone the `Detectron2` repository by running the following commands in your Colab notebook:

```bash
!pip install torch torchvision opencv-python rembg gfpgan onnxruntime
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
%cd detectron2_repo
!pip install -e .
%cd .. # Go back to the original directory if you want to save outputs there
```

*Note: `gfpgan` and `onnxruntime` are installed but not explicitly used in the provided code snippets for the core VTO logic. `gfpgan` is for face restoration and `onnxruntime` is for ONNX model inference, which might be useful for later deployment.*

-----

## Usage

This project is primarily designed to be run within a Google Colab notebook.

### 1\. Prepare Your Environment

1.  **Install Dependencies:** Run the installation commands provided in the "Libraries" section above.
2.  **Mount Google Drive:** This allows you to store your input images in Google Drive.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    You will be prompted to authorize Google Drive access.
3.  **Place Images:** Create a folder in your Google Drive (e.g., `My Drive/Colab Notebooks/VTO2/`) and place your `person.png` (image of a person) and `cloth.png` (image of the clothing item) inside it. Adjust the image paths in the training and inference sections to match your structure.

### 2\. Define Helper Functions and Network Architectures

Ensure you have the following code cells in your notebook:

  * **Image Utilities and Preprocessing:**

    ```python
    from PIL import Image
    from rembg import remove
    import cv2
    import numpy as np
    import torchvision.transforms as T
    import torch

    def load_image(path):
        return Image.open(path).convert("RGB")

    def remove_background_rgb(img):
        # Using rembg to remove background and ensure RGB output
        out = remove(img)
        out_rgb = Image.fromarray(np.array(out)).convert("RGB")
        return out_rgb

    def preprocess(img, size=(256, 192)):
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])
        return transform(img).unsqueeze(0)  # Shape: [1, 3, H, W]
    ```

  * **Person Segmentation with Detectron2:**

    ```python
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    def setup_segmentor():
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return DefaultPredictor(cfg)

    segmentor = setup_segmentor()

    def segment_person(image_np):
        # image_np should be a numpy array in BGR format (as expected by Detectron2)
        outputs = segmentor(image_np)
        # Assuming only one person instance and taking the first mask
        if outputs["instances"].has("pred_masks"):
            mask = outputs["instances"].pred_masks[0].cpu().numpy().astype(np.uint8)
            return Image.fromarray(mask * 255)
        return Image.fromarray(np.zeros(image_np.shape[:2], dtype=np.uint8)) # Return empty mask if no person found
    ```

    *Note: The `segment_person` function is defined but not directly used in the main training loop in the provided snippets. Its integration would typically involve using the generated mask to guide the warping or synthesis process.*

  * **Network Definitions (`GMM`, `TryOnSynthesizer`):**

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    class GMM(nn.Module):  # Simplified Thin Plate Spline proxy / Garment Motion Model
        def __init__(self):
            super(GMM, self).__init__()
            self.warp = nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1), # Output 3 channels for warped cloth
                nn.Tanh() # Tanh to scale output to [-1, 1] if desired, or ReLU for [0,1]
            )

        def forward(self, person, cloth):
            # Concatenate person and cloth images as input
            return self.warp(torch.cat([person, cloth], dim=1))

    class TryOnSynthesizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gen = nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1), # Output 3 channels for synthesized image
                nn.Tanh() # Tanh to scale output to [-1, 1] if desired, or ReLU for [0,1]
            )

        def forward(self, person, warped_cloth):
            # Concatenate person and warped cloth images as input
            return self.gen(torch.cat([person, warped_cloth], dim=1))
    ```

### 3\. Run the Training Loop

Execute the cell containing the training logic. This will load your images, remove their backgrounds, pass them through the networks, calculate loss, and update weights for a few epochs.

```python
gmm = GMM().cuda()
tom = TryOnSynthesizer().cuda()
optimizer = torch.optim.Adam(list(gmm.parameters()) + list(tom.parameters()), lr=0.001)

# Optional: Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(10):
    # Load and preprocess person image
    person_img_pil = load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/person.png')
    person_img = preprocess(person_img_pil).cuda()

    # Load, remove background, and preprocess cloth image
    cloth_img_pil = load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/cloth.png')
    cloth_img_no_bg = remove_background_rgb(cloth_img_pil)
    cloth_img = preprocess(cloth_img_no_bg).cuda()

    # Forward pass
    warped = gmm(person_img, cloth_img)
    output = tom(person_img, warped)

    # Calculate loss (simulated: ideally, you'd use a ground-truth try-on image here)
    loss = F.l1_loss(output, person_img) # Here, person_img is used as a placeholder target

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step the scheduler (if used)
    scheduler.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
```

### 4\. Save Model Checkpoint

```python
torch.save({
    'gmm': gmm.state_dict(),
    'tom': tom.state_dict()
}, 'virtual_tryon_model.pth')
```

### 5\. Run Inference and Visualize Result

```python
# Load the test person and garment images
test_person_pil = load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/person.png')
test_person = preprocess(test_person_pil).cuda()

test_cloth_pil = load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/cloth.png')
test_cloth_no_bg = remove_background_rgb(test_cloth_pil)
test_cloth = preprocess(test_cloth_no_bg).cuda()

# Set models to evaluation mode
gmm.eval()
tom.eval()

with torch.no_grad(): # Disable gradient calculations for inference
    warped_test = gmm(test_person, test_cloth)
    output_test = tom(test_person, warped_test)

# Visualize and save result
to_pil = T.ToPILImage()
# Squeeze batch dimension, move to CPU, and clamp pixel values to [0, 1]
result_image = output_test.squeeze(0).cpu().clamp(0, 1)
to_pil(result_image).save('test_tryon_result.png')
# to_pil(result_image).show() # .show() might not work directly in all Colab environments,
                            # you might need to display the image using matplotlib or download it.
```

-----

## Limitations and Future Improvements

While VTO 3 introduces crucial pre-processing steps, it's still a simplified model. For a production-ready or research-level virtual try-on system, consider these advancements:

  * **Sophisticated Warping:** The `GMM` is a placeholder for a true geometric matching or warping module. Real-world VTO systems would implement techniques like Thin Plate Spline (TPS) transformations guided by DensePose (human body keypoints and surface mapping), or advanced flow prediction networks to deform the garment realistically onto the person's body.
  * **Realistic Image Synthesis:** The `TryOnSynthesizer` uses simple convolutional layers. High-fidelity VTO requires more powerful generative models, such as GANs (Generative Adversarial Networks) or diffusion models, which can produce highly realistic textures, shadows, wrinkles, and seamlessly blend the warped garment with the person.
  * **Loss Functions:** Beyond L1 loss, employ a combination of:
      * **Perceptual Loss (VGG/LPIPS):** To encourage photo-realistic outputs.
      * **Adversarial Loss:** For GAN-based models to improve realism.
      * **Mask Loss/Segmentation Loss:** To ensure accurate placement and blending of the garment within the segmented person region.
      * **Smoothness Loss:** For the warping field to prevent unnatural deformations.
  * **Guided Synthesis:** Utilize the segmented person mask and potentially pose keypoints to explicitly guide the synthesizer on where and how to render the garment, ensuring it only appears on the body and not in the background.
  * **Occlusion Handling:** Implement mechanisms to handle cases where parts of the garment are occluded by the person's arms or other body parts, ensuring realistic rendering.
  * **High-Resolution Output:** Adapt models and training strategies for high-resolution image generation, which often involves progressive training or specialized upsampling layers.
  * **Dataset and Training:** Train on much larger and more diverse datasets with corresponding ground truth images (if available) for better generalization. Training would require significantly more computational resources (GPUs) and longer training times.
  * **Multi-Garment/Complex Garments:** Extend the model to handle different types of garments (dresses, jackets, etc.) and complex garment features.
  * **Interactive UI/API:** Develop a more robust API (e.g., with FastAPI) and potentially a user interface for a seamless try-on experience.

-----
