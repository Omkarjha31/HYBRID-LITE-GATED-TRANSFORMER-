# Hybrid Lite-Gated Transformer for Chest X-ray Classification

This project implements a hybrid deep learning model combining a **CNN (ResNet-18)** and a **Lite-Gated Transformer** for classifying chest X-ray images into:

- Normal  
- Pneumonia  

The model captures both **local spatial features** and **global contextual relationships**.

---

## Project Overview

Chest X-ray analysis is widely used for pneumonia detection. This project builds an automated system using a hybrid architecture that improves classification by combining CNN feature extraction with Transformer-based attention.

---

## Model Architecture

Input Image (224×224)  
→ ResNet-18 (CNN Backbone)  
→ Feature Maps → Tokenization  
→ Linear Projection + CLS Token  
→ Lite-Gated Transformer  
→ Fully Connected Layer  
→ Prediction (Normal / Pneumonia)

---

## Repository Structure

├── hybrid_lgt_cnn.py      # Main training code (CNN + Transformer)  
├── ablation_study.py      # Ablation study code  
├── README.md  

---

## Dataset

- Total Images: 5,863  
- Classes: Normal, Pneumonia  
- Structure: train / val / test folders  

---

## Training Configuration

- Framework: PyTorch  
- Image Size: 224 × 224  
- Optimizer: Adam  
- Learning Rate: 1e-4  
- Epochs: 30  
- Batch Size: 16  
- Early Stopping: Patience = 5  

---

## Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | 88%    |
| Precision  | 0.90   |
| Recall     | 0.88   |
| F1 Score   | 0.88   |
| ROC-AUC    | 0.9688 |

### Key Observations

- Pneumonia Recall ≈ 0.99 (very high detection rate)  
- Some false positives (acceptable in medical screening)  
- Strong class separability (high ROC-AUC)  

---

## Ablation Study

| Model      | Accuracy |
|------------|----------|
| LGT + CNN  | 0.98     |
| LGT Only   | 0.56     |
| No Gate    | 0.99     |

### Insights

- CNN backbone is critical for performance  
- Transformer alone performs poorly  
- Gating mechanism does not significantly change accuracy  

---

## How to Run

1. Clone repository  
git clone <your-repo-link>  
cd <repo-name>  

2. Install dependencies  
pip install torch torchvision scikit-learn matplotlib  

3. Run main model  
python hybrid_lgt_cnn.py  

4. Run ablation study  
python ablation_study.py  

---

## Outputs

Running the code will generate:

- Accuracy curve  
- Loss curve  
- F1 score curve  
- ROC curve  
- Confusion matrix  
- Ablation results  

---

## Limitations

- Small validation set causes metric fluctuations  
- Higher false positives for Normal class  
- Gating impact not clearly visible in accuracy  

---

## Future Work

- Improve class balance handling  
- Apply threshold tuning  
- Use larger datasets  
- Add interpretability methods (Grad-CAM)  

---

## Author

Omkar Jha  
UPES | SAP ID: 500119757  

---

## If you found this useful, consider giving a star!