# VTO 2: Virtual Try-On with PyTorch (Simple Neural Network Approach)

This project, "VTO 2," explores a more sophisticated approach to Virtual Try-On (VTO) compared to the basic image blending in VTO 1. It utilizes a simple neural network architecture in PyTorch to attempt a "warping" of clothing onto a person's image and then synthesizes the final try-on result. This serves as a foundational step towards more complex, generative VTO models.

-----

## Features

  * **PyTorch-based Implementation:** Built entirely using PyTorch for defining, training, and inferencing neural networks.
  * **Modular Network Design:** Separates the try-on process into two main components:
      * `SimpleWarpNet`: A conceptual network aimed at warping the product (cloth) image based on the person's image. (In this simplified POC, it generates an output that is then fed to the synthesizer, but a true warp network would predict flow fields or transformations).
      * `TryOnSynthesizer`: A network that takes the person's image and the (conceptually) warped cloth image to generate the final try-on output.
  * **Basic Training Loop:** Demonstrates a fundamental training process, loading images, performing forward passes, calculating a simple L1 loss, and optimizing the network parameters.
  * **Image Preprocessing:** Includes utilities for loading and preprocessing images using `torchvision.transforms`.
  * **Google Drive Integration:** Shows how to mount Google Drive to easily access input images.
  * **Loss Visualization:** Plots the training loss over epochs to show the model's learning progression.
  * **Model Saving:** Saves the trained model's state dictionary for future use.

-----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * Google Colab (recommended for easy GPU access and file handling)

### Libraries

You can install the necessary Python libraries using pip:

```bash
pip install torchvision opencv-python matplotlib albumentations pillow
```

*Note: `albumentations` is installed but not used in the provided code snippet.*

-----

## Usage

This project is designed to be run within a Google Colab notebook.

### 1\. Prepare Your Environment

1.  **Install Dependencies:** Run the first cell in your Colab notebook:
    ```python
    !pip install torchvision opencv-python matplotlib albumentations
    ```
2.  **Mount Google Drive:** This allows you to store your `person.png` and `cloth.png` files in Google Drive.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    You will be prompted to authorize Google Drive access.
3.  **Place Images:** Create a folder in your Google Drive (e.g., `My Drive/Colab Notebooks/VTO2/`) and place your `person.png` (image of a person) and `cloth.png` (image of the clothing item) inside it. Adjust the paths in the training loop accordingly.

### 2\. Define Helper Functions and Network Architectures

Ensure you have the following code cells in your notebook:

  * **Image Loading and Preprocessing:**

    ```python
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image

    def load_image(path):
        img = Image.open(path).convert("RGB")
        return img

    def preprocess(img, size=(256, 192)):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        return transform(img).unsqueeze(0)  # (1, C, H, W)
    ```

  * **Fake Mask Creation (for conceptual use):**

    ```python
    def create_fake_mask(image_tensor):
        # Simulate a mask for demo purpose
        mask = torch.zeros_like(image_tensor)
        mask[:, :, 64:192, 32:160] = 1  # crude human-like body region
        return mask
    ```

  * **Network Definitions (`SimpleWarpNet`, `TryOnSynthesizer`):**

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleWarpNet(nn.Module):
        def __init__(self):
            super(SimpleWarpNet, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh()
            )

        def forward(self, person, cloth):
            x = torch.cat([person, cloth], dim=1) # Concatenate person and cloth images
            return self.conv(x)

    class TryOnSynthesizer(nn.Module):
        def __init__(self):
            super(TryOnSynthesizer, self).__init__()
            self.generator = nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh()
            )

        def forward(self, person, warped_cloth):
            x = torch.cat([person, warped_cloth], dim=1) # Concatenate person and (conceptually) warped cloth
            return self.generator(x)
    ```

### 3\. Run the Training Loop

Execute the cell containing the training logic. This will load your images, pass them through the networks, calculate loss, and update weights for a few epochs.

```python
warp_net = SimpleWarpNet().cuda()
tryon_net = TryOnSynthesizer().cuda()
optimizer = torch.optim.Adam(list(warp_net.parameters()) + list(tryon_net.parameters()), lr=0.001)

for epoch in range(10):
    person_img = preprocess(load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/person.png')).cuda()
    cloth_img = preprocess(load_image('/content/drive/MyDrive/Colab Notebooks/VTO2/cloth.png')).cuda()

    # The unsqueeze(0) below adds a batch dimension if not already present
    # It seems preprocess already adds one, so this might be redundant if preprocess is consistent.
    # Keep it for safety if your images might sometimes be loaded without a batch dim.
    person_img = person_img.unsqueeze(0).cuda() if person_img.dim() == 3 else person_img.cuda()
    cloth_img = cloth_img.unsqueeze(0).cuda() if cloth_img.dim() == 3 else cloth_img.cuda()


    warped_cloth = warp_net(person_img, cloth_img)
    output = tryon_net(person_img, warped_cloth)

    loss = F.l1_loss(output, person_img)  # Simulated: Ideally, you'd use a ground-truth try-on image here
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
```

### 4\. Visualize Loss

Run the cell to display the training loss plot:

```python
import matplotlib.pyplot as plt

loss_values = [0.7574, 0.6781, 0.5994, 0.5143, 0.4211, 0.3259, 0.2502, 0.1855, 0.1387, 0.1190]
plt.plot(loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()
```

### 5\. Save and Display Result

Finally, save the generated try-on image and attempt to display it:

```python
import torchvision.transforms as T

to_pil = T.ToPILImage()
result_image = output.squeeze(0).cpu().clamp(0, 1)
to_pil(result_image).save("tryon_result.png")
# to_pil(result_image).show() # .show() might not work directly in all Colab environments
```

### 6\. Save Model Checkpoint

```python
torch.save({
    'warp_net': warp_net.state_dict(),
    'tryon_net': tryon_net.state_dict()
}, 'virtual_tryon_model.pth')
```

-----

## Project Structure (Conceptual)

```
VTO 2/
├── README.md              (This file)
├── vto2_notebook.ipynb    (Your Google Colab notebook containing the code)
├── person.png             (Example input: image of a person - to be placed in Google Drive)
└── cloth.png              (Example input: image of clothing - to be placed in Google Drive)
└── tryon_result.png       (Output image from the try-on process)
└── virtual_tryon_model.pth (Saved model weights)
```

-----

## Limitations and Future Improvements

This project is a very simplified demonstration of a neural network-based virtual try-on. Significant improvements are needed for a realistic application:

  * **Advanced Warping:** The `SimpleWarpNet` is rudimentary. Real VTO systems employ sophisticated warping modules (e.g., using DensePose, affine transformations, or optical flow prediction) to precisely align clothing to the person's body shape and pose.
  * **Realistic Synthesis:** The `TryOnSynthesizer` performs a simple concatenation and convolution. State-of-the-art VTO models often use Generative Adversarial Networks (GANs) or diffusion models to produce highly realistic and high-resolution try-on images, handling details like wrinkles, shadows, and fabric textures.
  * **Loss Functions:** The current `L1_loss` is basic. Realistic VTO training typically involves:
      * **Perceptual Loss (VGG/LPIPS):** To ensure visual similarity beyond pixel-wise differences.
      * **GAN Loss:** For generating realistic images that are indistinguishable from real photos.
      * **Style Loss:** To match the style of the target clothing.
      * **Mask Loss:** If segmentation masks are used for different body parts.
  * **Data and Training:**
      * **Dataset:** Training requires large datasets of person-clothing pairs, often with associated segmentation masks, pose keypoints, and ground-truth try-on images.
      * **Longer Training:** Real VTO models are trained for many more epochs on GPUs for days or weeks.
      * **Complex Architectures:** Incorporating U-Net structures, attention mechanisms, and various normalization layers is crucial for performance.
  * **Segmentation and Pose Estimation:** For precise try-on, accurate human parsing (segmentation of body parts) and pose estimation are essential pre-processing steps.
  * **Occlusion Handling:** The current model doesn't explicitly handle occlusions (e.g., if a hand covers part of the clothing). Advanced models include mechanisms to render occluded parts realistically.
  * **Resolution:** The current implementation processes small image sizes. High-resolution VTO is a major challenge that requires specialized architectures.

This project serves as a starting point to understand the very basic components of a deep learning-based virtual try-on system. For practical applications, research into existing VTO papers and open-source implementations (e.g., VITON-HD, CP-VTON+, M\&M VTO) is recommended.
