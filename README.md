# Waste Classification Using Deep Learning

Image classifier that sorts waste into six categories (cardboard, glass, metal, paper, plastic, trash) using transfer learning with MobileNetV2.

## Project links
- Live demo (Vercel): [add link]
- Project report: [add link]
- Presentation video (YouTube): [add link]
- LinkedIn post: [add link]

## Requirements checklist
| Requirement | Where |
|---|---|
| Dataset | Kaggle "Garbage Classification" (kagglehub slug: asdasdasasdas/garbage-classification); sample images in `samples/` |
| Trained model weights | `weights/waste_model.pth` (PyTorch), `weights/waste_model.onnx` (web) |
| Source code | `train_waste_classifier.ipynb`, `predict.py` |
| Web deployment | Vercel link above |
| Report, slides, video | Links above |

## Dataset
Secondary dataset from Kaggle with 2,527 images in six classes. Split 70/15/15 with seed 42: 1,768 train, 379 validation, 380 test images.

## Model and training
- Compared MobileNetV2, ResNet18 and EfficientNet-B0, all pretrained on ImageNet and fully fine-tuned.
- Input 224x224, augmentation (flip, rotation, color jitter), AdamW (lr 3e-4), cosine LR schedule, label smoothing 0.1, early stopping (patience 3), best checkpoint kept.
- Selected model: MobileNetV2 (highest validation accuracy; the three models performed comparably).

## Results
| Model | Validation accuracy |
|---|---|
| MobileNetV2 | 94.46% |
| ResNet18 | 92.35% |
| EfficientNet-B0 | 93.93% |

**Test accuracy (MobileNetV2): 93.16%** (354 of 380 held-out images). Per-class results are in `results/classification_report.txt`; the confusion matrix, training curves and Grad-CAM are in `results/`.

## How to run
```
pip install -r requirements.txt
python predict.py samples/<any image>.jpg
```

## Limitations
- Small test set (380 images), so the accuracy estimate has a margin of about 2.5 points.
- Dataset photos use plain backgrounds; real-world photos may score lower.
- "Trash" is the smallest and weakest class (F1 0.81). Plastic is often confused with glass and metal.
- Mild overfitting (training accuracy about 99%, validation about 94%).
