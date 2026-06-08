# 🔍 Leather Defect Detection & Localization using YOLO

An AI-powered Computer Vision application for automated leather defect detection and localization using YOLO object detection models. This system identifies defects on leather surfaces and highlights them with bounding boxes, enabling efficient quality inspection for manufacturing environments.

---

## 🌐 Live Demo

**Streamlit Application:**
https://leather-defect-detection.streamlit.app/

---

## 📌 Overview

Leather quality inspection is an essential step in leather manufacturing and product development. Traditional manual inspection methods are time-consuming, labor-intensive, and susceptible to human error.

This project leverages YOLO (You Only Look Once) object detection models to automate the process of defect identification and localization. The model analyzes leather images, detects defects, generates bounding boxes around defect regions, and provides confidence scores for each prediction.

The application is deployed using Streamlit, allowing users to upload images and obtain real-time detection results through a simple web interface.

---

## 🎯 Problem Statement

In industrial leather manufacturing, defects such as scratches, fold marks, holes, cuts, and surface imperfections reduce product quality and increase production costs.

Challenges with manual inspection include:

* High dependency on skilled inspectors
* Inconsistent defect identification
* Time-consuming inspection process
* Difficulty in scaling inspection workflows
* Increased operational costs

The objective of this project is to automate defect detection using Deep Learning and Computer Vision techniques.

---

## 🚀 Key Features

### AI-Powered Defect Detection

* YOLO-based object detection
* Real-time image inference
* Multi-class defect classification
* Accurate defect localization
* Confidence score generation

### Interactive Web Application

* Upload leather images
* View original image
* Visualize detected defects
* Download processed results
* Defect summary generation

### Industrial Use Case

* Automated quality inspection
* Manufacturing process optimization
* Defect analysis and reporting
* Reduced manual inspection effort

---

## 🧠 Machine Learning & Computer Vision Concepts

This project incorporates several important AI concepts:

### Computer Vision

* Object Detection
* Image Processing
* Defect Localization
* Bounding Box Prediction

### Deep Learning

* Convolutional Neural Networks (CNNs)
* Transfer Learning
* Model Fine-Tuning

### Data Engineering

* Dataset Preparation
* Data Annotation
* Data Augmentation
* Dataset Splitting

### Model Evaluation

* Confidence Scores
* Detection Accuracy
* Inference Analysis

---

## 🏗️ System Architecture

```text
                ┌────────────────────┐
                │   Input Image      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Streamlit Web App  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ YOLO Detection     │
                │ Model              │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Defect Detection   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Localization       │
                │ Bounding Boxes     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Confidence Scores  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Detection Results  │
                └────────────────────┘
```

---

## 🛠️ Technology Stack

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning Framework | PyTorch                   |
| Object Detection        | YOLO                      |
| Computer Vision         | OpenCV                    |
| Data Processing         | NumPy, Pandas             |
| Image Handling          | Pillow                    |
| Web Interface           | Streamlit                 |
| Version Control         | Git, GitHub               |
| Deployment              | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
LeatherDefect/
│
├── app.py                      # Streamlit application
├── modelv1.pt                  # Trained YOLO model
├── requirements.txt            # Project dependencies
│
├── LEATHERDATASETv1/           # Original dataset
├── AUGMENTED_IMAGESv2/         # Augmented images
├── YOLODATASET/                # YOLO formatted dataset
├── YOLODATASETv2/
├── YOLODATASETv3/
│
├── Newdefects/                 # Additional defect samples
├── testdataset/                # Test images
├── runs/                       # YOLO prediction outputs
│
├── annotate.py                 # Annotation utilities
├── image_augment.py            # Data augmentation
├── image_format_convert.py     # Image preprocessing
├── rollinspection.py           # Inspection pipeline
├── testing.py                  # Model testing
├── test_model.py               # Inference script
└── README.md
```

---

## 🔬 Dataset Preparation

The dataset preparation pipeline involved:

### Data Collection

* Leather surface images
* Multiple defect categories
* Industrial inspection samples

### Annotation

Images were manually annotated using bounding boxes to indicate defect regions.

### Data Augmentation

The following augmentation techniques were used:

* Rotation
* Flipping
* Scaling
* Image Transformations

These techniques improve model generalization and reduce overfitting.

---

## 🤖 Model Information

### Algorithm

YOLO Object Detection

### Framework

Ultralytics YOLO

### Backend

PyTorch

### Model Output

For every detected defect, the model predicts:

* Defect Class
* Bounding Box Coordinates
* Confidence Score

Example:

```text
Class: FoldMark
Confidence: 98.21%
```

---

## 📷 Application Workflow

### Step 1

Upload a leather image through the Streamlit interface.

### Step 2

The image is processed by the trained YOLO model.

### Step 3

Defects are identified and localized.

### Step 4

Bounding boxes are drawn around detected defects.

### Step 5

Confidence scores are generated.

### Step 6

Detection results are displayed and can be downloaded.

---

## 📊 Sample Output

```text
Detected 1 defect(s)

Defect Summary

✓ FoldMark : 1

Confidence Scores

FoldMark      98.21%
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Akash2027/LeatherDefect.git
```

### Navigate to Project Directory

```bash
cd LeatherDefect
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python -m streamlit run app.py
```

---

## 🌍 Deployment

The application is deployed using Streamlit Community Cloud.

Live Application:

https://leather-defect-detection.streamlit.app/

---

## 📈 Future Enhancements

Potential improvements include:

* Real-time video defect detection
* Webcam-based inspection
* Batch image processing
* Defect analytics dashboard
* Docker containerization
* Kubernetes deployment
* MLOps integration
* Cloud deployment on AWS/Azure/GCP
* Automated model retraining pipelines

---

## 🎓 Skills Demonstrated

This project demonstrates practical experience in:

* Python Programming
* Deep Learning
* Computer Vision
* Object Detection
* YOLO
* PyTorch
* OpenCV
* Streamlit
* Data Augmentation
* Dataset Annotation
* Model Deployment
* Git & GitHub
* End-to-End AI Application Development

---

## 👨‍💻 Author

**Akash K**

M.Tech Software Engineering

GitHub: https://github.com/Akash2027

LinkedIn: www.linkedin.com/in/akash-k-bb9a20274

---

## ⭐ Support

If you found this project useful, consider giving the repository a star.

Your support helps improve and maintain future AI and Computer Vision projects.
