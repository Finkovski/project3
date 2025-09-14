# 🚀 Project 2 - CIFAR-10 Image Classification (Group 2)

## ✨ Project Overview
This project focuses on building and evaluating models for CIFAR-10 image classification.

We developed and compared two main approaches:
- A **Custom Convolutional Neural Network (CNN)**
- **Transfer Learning** using **MobileNetV2** pretrained on ImageNet.

The best performing model (MobileNetV2) is fully trained, fine-tuned, evaluated, and deployed.

---

## 🗂 Repository Structure
```
Project2_G2.ipynb                 # Main Notebook: final chosen model (MobileNetV2)
Project2_G2_other_models.ipynb     # Additional experiments: other CNNs and architectures
requirements.txt                   # Required libraries
REPORTmd.md                        # Report (Markdown)
REPORTpdf.pdf                      # Report (PDF)
README.md                          # This file
```

---

## 📦 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Finkovski/Project2-G2.git
cd Project2-G2
```
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Open `Project2_G2.ipynb` and run the notebook.

---

## 📚 Models Included

### Final Chosen Model: MobileNetV2
- Transfer learning approach
- Top layers customized
- Fine-tuned top 50 layers
- Achieved ~90% accuracy on CIFAR-10

### Other Models
- Several custom CNN architectures
- Experiments included in `Project2_G2_other_models.ipynb`

---

## 📈 Evaluation
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 🌍 Deployment
- A basic Gradio web app allows users to upload an image and receive classification predictions.

---

# ✨ Thank you!