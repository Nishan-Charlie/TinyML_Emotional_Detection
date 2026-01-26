# LightCBAMNet: A Tiny Deep Learning Model for Emotion Detection on Edge Devices

LightCBAMNet is a lightweight, high-performance convolutional neural network designed specifically for Facial Emotion Recognition (FER) in resource-constrained environments. By combining efficient depthwise convolutions with a specialized attention mechanism, this model achieves a balance between low parameter counts and high classification accuracy.

---

## 🔬 Research Focus
This research addresses the challenge of deploying robust Emotion Detection on small edge devices (e.g., IoT sensors, mobile devices, and micro-controllers). 

### Key Innovations:
* **Phish Activation**: Uses the $x \cdot \tanh(\text{GELU}(x))$ activation function for smoother gradient flow compared to standard ReLU.
* **DCWP Module**: Implements Depthwise Convolution with Phish activation to minimize FLOPs (Floating Point Operations) while maintaining feature richness.
* **CBAM Integration**: A Convolutional Block Attention Module that sequentially applies Channel and Spatial attention, allowing the model to focus on critical facial landmarks like the eyes, nose, and mouth.
* **Efficiency**: Optimized for a tiny memory footprint, making it ideal for real-time inference on hardware without dedicated GPUs.



---

## 🏗️ Architecture Breakdown
The model follows a streamlined, modular pipeline:

1.  **Stem**: A large-kernel (7x7) convolution with stride 3 for rapid spatial downsampling.
2.  **DCWP Layers**: Two stages of depthwise-pointwise convolutions for efficient feature extraction.
3.  **LR Modules**: Linear Bottleneck Residual modules to refine features and prevent vanishing gradients.
4.  **Attention Head**: The **CBAM** block (Channel Attention followed by Spatial Attention) to weight important features.
5.  **Classifier**: Global Average Pooling followed by Dropout and a Linear layer for final emotion classification.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 1.7+
* Torchvision
* Scikit-learn

### Installation
```bash
pip install torch torchvision scikit-learn tqdm