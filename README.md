# 🎨 Neural Style Transfer using VGG19 and TensorFlow

## Overview

This project implements **Neural Style Transfer (NST)** using a pretrained **VGG19 convolutional neural network** in TensorFlow and Keras.

The goal of neural style transfer is to generate a new image that preserves:
- the **content** of one image,
- while adopting the **artistic style** of another image.

The project is based on the seminal paper:

> *A Neural Algorithm of Artistic Style*  
> Leon Gatys, Alexander Ecker, Matthias Bethge (2015)

using feature representations extracted from a pretrained VGG19 network.

---

# 🧠 Core Idea

Neural Style Transfer works by optimizing a generated image such that:

- its high-level feature representations match the **content image**
- while its feature correlations match the **style image**

The optimization process uses activations from different layers of a pretrained CNN.

---

# 🖼️ Style Transfer Pipeline

The project combines three images:

| Image Type | Purpose |
|---|---|
| Content Image | Preserves structure and objects |
| Style Image | Provides artistic texture and appearance |
| Generated Image | Optimized result combining both |

The generated image is iteratively updated using gradient-based optimization until both:
- content loss,
- and style loss

are minimized.

---

# 🏗️ Model Architecture

This implementation uses the pretrained **VGG19** convolutional neural network.

## Why VGG19?

VGG19 is widely used for neural style transfer because:
- convolutional layers learn hierarchical visual representations,
- shallow layers capture textures and colors,
- deeper layers capture semantic content and structures.

The network weights were pretrained on the **ImageNet** dataset.

For feature extraction, VGG19 was loaded using:

- `include_top = False`
- pretrained ImageNet weights
- intermediate convolutional layer outputs

This allowed the project to extract feature representations directly from convolutional layers without using the classification head.

---

# 🔍 Feature Extraction

A custom **feature extractor** was implemented using selected convolutional layers from VGG19.

The feature extractor computes:

- Content representations
- Style representations
- Gram matrices

from intermediate CNN activations.

## Content Representation

High-level CNN activations were used to preserve:
- object structure,
- spatial arrangement,
- semantic information.

The content loss was computed using feature activations from selected VGG19 layers:

```math
J_{content}(x,c)=\frac{1}{2|L_{content}|}\sum_{\ell \in L_{content}} ||a^{[\ell](x)} - a^{[\ell](c)}||_2^2
```

where:
- \(a^{[\ell](x)}\) represents feature activations of the generated image,
- \(a^{[\ell](c)}\) represents feature activations of the content image.

---

## Style Representation

Style was captured using **Gram Matrices**, which compute correlations between CNN feature maps.

Gram matrices enable the model to learn:
- textures,
- brush strokes,
- color distributions,
- artistic patterns.

The style loss was computed as:

```math
J_{style}(x,s)=\frac{1}{4|L_{style}|}\sum_{\ell \in L_{style}} ||G^{[\ell](x)} - G^{[\ell](s)}||_2^2
```

where:
- \(G^{[\ell](x)}\) is the Gram matrix of the generated image,
- \(G^{[\ell](s)}\) is the Gram matrix of the style image.

The Gram matrix was computed using feature channel correlations:

```math
G_{i,j}^{[\ell](x)}=
\frac{1}{n_H^{[\ell]}n_W^{[\ell]}n_C^{[\ell]}}
\langle a_i^{[\ell](x)}, a_j^{[\ell](x)} \rangle
```

This allows style information to be represented independently of exact spatial arrangement.

---

# 📐 Loss Functions

The final optimization objective combined content and style losses:

```math
L_{total} = \alpha L_{content} + \beta L_{style}
```

where:
- α controls content preservation,
- β controls artistic stylization.

---

# 🌊 Total Variation (TV) Regularization

To improve image smoothness and reduce high-frequency noise, **Total Variation (TV) regularization** was added to the optimization process.

Neural style transfer often produces images with:
- noisy textures,
- sharp pixel-level artifacts,
- excessive local variations.

TV regularization penalizes strong local intensity differences, encouraging smoother generated images.

TensorFlow’s built-in total variation function was incorporated directly into the custom training loop.

## Benefits of TV Regularization

- Reduced image noise
- Smoother textures
- Improved visual coherence
- Better artistic appearance

This significantly improved the perceptual quality of generated images.

---

# ⚙️ TensorFlow & Keras Concepts Used

This project involved advanced TensorFlow and Keras functionality including:

- Keras Functional API
- Keras Subclassing API
- TensorFlow custom training loops
- Automatic differentiation
- Gradient computation with `tf.GradientTape`
- Optimization of image tensors directly
- Feature extraction from pretrained networks
- Gram matrix computation
- TV regularization

Unlike traditional deep learning projects, this work does **not train a neural network**. Instead, the generated image itself is optimized.

---

# 🚀 Optimization Strategy

The generated image was initialized and iteratively optimized using gradient descent.

## Optimization Workflow

1. Load content and style images
2. Pass images through pretrained VGG19
3. Extract intermediate feature representations
4. Compute content and style losses
5. Compute Gram matrices
6. Apply TV regularization
7. Compute gradients using automatic differentiation
8. Update generated image
9. Repeat optimization until convergence

---

# 🎨 Results

The generated images successfully:
- preserved semantic structure from the content image,
- while transferring artistic textures and color patterns from the style image.

The experiments demonstrated how pretrained CNN feature representations can separate:
- content information,
- and style information

within deep neural networks.

The addition of TV regularization further improved:
- image smoothness,
- texture consistency,
- and perceptual quality.

---

# 🔬 Key Insights

This project demonstrates several important deep learning concepts:

- CNNs learn hierarchical visual features
- Different CNN layers encode different types of information
- Style can be represented through feature correlations
- Pretrained networks can be repurposed for generative tasks
- Optimization can be applied directly to image pixels
- Gram matrices capture artistic texture information
- Regularization improves perceptual image quality

The project highlights the expressive power of convolutional feature representations in computer vision.

---


# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL / OpenCV
- Jupyter Notebook

---

# 📂 Repository Structure

```bash
├── notebooks/
│   └── Neural_Style_Transfer.ipynb
├── images/
│   ├── content/
│   ├── style/
│   └── generated/
├── figures/
├── README.md
└── requirements.txt
```

---

# 🔮 Future Improvements

Potential future extensions include:

- Fast Neural Style Transfer
- Real-time style transfer
- Arbitrary style transfer
- Style transfer using transformers
- Multi-style blending
- Video style transfer
- Diffusion-based stylization methods

---

# 📌 Conclusion

This project implemented neural style transfer using pretrained convolutional neural networks and optimization-based image synthesis.

The work demonstrates how deep CNN feature representations can separate content and artistic style, enabling the creation of visually compelling generated images.

The project also provided hands-on experience with advanced TensorFlow workflows, custom optimization pipelines, and feature-based image generation techniques.

---

# 👨‍💻 Author

**Raj Mahadevwala**

Master’s Student – Data Science / AI  
Technical University of Braunschweig

---
