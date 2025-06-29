# 🧾 Cervical Cancer Classification Project

---

## 📁 Data and Data Collection

### Dataset Source

- The dataset used contains labeled images of cervical cell types (Type 1, Type 2, and Type 3).
  ...or you can state if there is a specific source you got it from...

### Dataset Size

- Include total number of images used (7675).
- Mention the distribution across the three classes (how many images per class).

### Data Type

- Image data (.jpg, .png).
- Image resolution is standardized to 224x224 pixels during processing.

### Data Preprocessing

- All images were resized to 224x224 pixels.
- Normalization applied: pixel values scaled to the range 0 to 1.
- Images reshaped for model input.

---

## 🧠 Deep Learning Methodology

### Transfer Learning and Pretrained Model

- A **Convolutional Neural Network (CNN)** was used via **transfer learning**.
- Pretrained base model used: **ResNet50V2**.
- Only the top layers were retrained on the cervical cancer dataset for the first traininig round.
- The other layers (convolution layers) were fine-tuned in the second round.

### Why Transfer Learning Was Used

- Speeds up training by using pretrained weights from large-scale datasets (like ImageNet).
- Reduces overfitting due to limited size of the medical dataset.
- Leverages learned visual features from general image data.

### Model Architecture

- Pretrained CNN base (frozen or partially trainable).
- Additional layers: Dense (fully connected), Dropout (to prevent overfitting), and Softmax (for multi-class classification).

### Training Process

- Dataset was split into **training** and **validation/test** sets.
- Model was trained over several epochs.
- Performance was evaluated using validation accuracy and loss.
- Callbacks were used. Check `modeling.ipynb` for the callbacks and search meaning and effects.

### Optimizer and Loss Function

- Optimizer: **Adam**
- Loss function: **categorical cross-entropy**
- Evaluation metric: **accuracy**.

---

## 📊 Performance Metrics and Model Evaluation

### Metrics to Report

- **Training accuracy**
- **Validation accuracy**
- **Training loss**
- **Validation loss**
- **Test accuracy**

### How to Find Metrics

- Check the training logs in the Jupyter Notebook (`modeling.ipynb`).
- Plots of accuracy/loss over epochs (can be added in Chapter 4).
- Final evaluation score printed by `model.evaluate(...)`.

### Important Visualization

- Confusion matrix: shows how many predictions were correct for each class.
- Accuracy/Loss curve: visualizes model performance over training.

---

## 🧱 System Design and Architecture

### Modules Overview

1. **Data Processing Module**

   - Handles image preprocessing: resizing, normalization, reshaping.
   - Uses `img_to_array`, `resize`, and `np.expand_dims`.

2. **Model Module**

   - Loads `.keras` model file.
   - Performs prediction on preprocessed image data.

3. **Web Interface (Frontend)**
   - Built using **Streamlit**.
   - Allows users to upload an image via the browser.
   - Displays:
     - Uploaded image.
     - Predicted class.
     - Confidence level (%).

### Technologies Used

- **Python**
- **TensorFlow / Keras** (deep learning framework)
- **Streamlit** (for web interface)
- **PIL and NumPy** (image and numerical processing)

---

## 🖼 Web Interface Features

### Functionalities

- Upload cervical image (`.jpg`, `.jpeg`, `.png`).
- Shows uploaded image preview.
- Automatically processes and classifies the image.
- Displays prediction and confidence.

### User Interaction

- Simple UI: One-click image upload.
- No setup required (Streamlit app can run locally).

---

## 🛠 How to Navigate the Files

### `modeling.ipynb`

- Contains model training process and performance metrics.
- Use outputs, accuracy/loss graphs, and printed evaluation metrics for Chapter 4.

### `app.py`

- Contains code for the web interface.
- Explains the model loading, image preprocessing, and prediction steps.
- Shows how the interface is built and interacts with the model.

### `models/cervical_cancer_classifier.keras`

- Saved model file used for prediction.

---
