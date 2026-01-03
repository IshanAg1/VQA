# 👁️ Visual Core: Advanced Visual Question Answering System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Visual Core** is a cutting-edge multimodal Artificial Intelligence system designed to bridge the gap between Computer Vision and Natural Language Processing. It allows users to query images using natural language and receive precise, context-aware answers in real-time.

---

## 📸 Project Demonstration
*(Add a screenshot of your running application here)*
![App Screenshot](assets/app_screenshot.png)

---

## 🧠 System Architecture & Methodology

The core of this project relies on a **Multimodal Fusion Architecture**. Unlike traditional AI that handles either text or images, VQA requires the simultaneous processing and understanding of both.

### 1. The Visual Pipeline (Convolutional Neural Networks)
To "see" the image, we utilize **ResNet50 (Residual Network)**, a deep convolutional neural network trained on the ImageNet dataset.
* **Function:** It processes the raw pixel data and extracts a high-dimensional feature map (2048-dimensional vectors).
* **Why ResNet?** It solves the vanishing gradient problem, allowing for deeper networks that capture intricate details like texture, shape, and spatial relationships.

![ResNet Architecture](assets/resnet.png)


### 2. The Textual Pipeline (Transformer Models)
To "read" the question, we employ **BERT (Bidirectional Encoder Representations from Transformers)**.
* **Function:** It tokenizes the input question and converts it into dense word embeddings.
* **Why BERT?** Unlike directional models (RNNs), BERT reads the entire sentence at once, allowing it to understand context and nuance (e.g., distinguishing "bank" of a river from "bank" for money).

![BERT Architecture](assets/bert.png)


### 3. Multimodal Fusion (The "Brain")
This is the critical phase where vision meets language.
1.  **Feature Projection:** Both the image features (from ResNet) and text features (from BERT) are projected into a shared embedding space.
2.  **Fusion Mechanism:** We utilize element-wise multiplication (Hadamard product) or concatenation to fuse the vectors.
3.  **Classification:** The fused vector is passed through a fully connected layer (Classifier) to predict the most probable answer from the vocabulary.

![VQA Architecture](assets/vqa_arch.png)


---

## 🚀 Key Features

* **State-of-the-Art Accuracy:** Powered by the ViLT (Vision-and-Language Transformer) architecture, achieving high accuracy on standard benchmarks.
* **Real-Time Inference:** Optimized for low-latency performance, delivering answers in milliseconds.
* **Top-5 Probability Distribution:** The system provides transparency by displaying the top 5 likely answers with their confidence scores.
* **Cyber-Aesthetic UI:** A custom-designed, dark-themed interface built with Streamlit for a modern, professional user experience.
* **Robust Error Handling:** Includes comprehensive error checking for file uploads and model initialization.

---

## 🛠️ Technical Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch
* **Model Hub:** Hugging Face Transformers
* **Image Processing:** Pillow (PIL)
* **Frontend Framework:** Streamlit
* **Version Control:** Git

---

## 📂 Directory Structure

```plaintext
Visual-Core-VQA/
├── assets/                  # Stores images for README and UI assets
│   ├── resnet.png
│   ├── bert.png
│   └── vqa_arch.png
├── app.py                   # The main entry point for the Streamlit application
├── requirements.txt         # List of all python dependencies
├── README.md                # Project documentation
└── .gitignore               # Configuration to ignore unnecessary files
