Pneumonia Detection via CNN – 94% Test Accuracy

This project uses a deep learning model (CNN) to detect pneumonia from chest X-ray images. It takes X-ray images and classifies them as either Pneumonia or Normal with about 93% accuracy.

We combined data from multiple sources, cleaned and prepared it, and trained a model that can help doctors or medical researchers quickly and accurately identify pneumonia. This can save time and improve diagnosis, especially in areas with limited medical staff.
## Dataset Information


This project utilizes chest X-ray image datasets from Kaggle. Specifically, the data is sourced from:

1.  **Chest X-Ray Images (Pneumonia) - COVID19, Pneumonia, Normal**: This dataset is used for the initial loading and provides a base for pneumonia and normal cases, excluding COVID-19.
    * **Source:** `/kaggle/input/chest-xray-covid19-pneumonia/Data/`
    * **Shapes after loading and preprocessing (x_train1, y_train1, x_test1, y_test1):**
        * Training Data: (4684, 150, 150, 1)
        * Training Labels: (4684,)
        * Test Data: (1172, 150, 150, 1)
        * Test Labels: (1172,)

2.  **Chest X-Ray Images (Pneumonia)**: Another standard dataset for pneumonia detection.
    * **Source:** `/kaggle/input/chest-xray-pneumonia/chest_xray/`
    * **Shapes after loading and preprocessing (x_train, y_train, x_val, y_val, x_test, y_test before combination):**
        * Training Data: (5216, 150, 150, 1)
        * Training Labels: (5216,)
        * Validation Data: (16, 150, 150, 1)
        * Validation Labels: (16,)
        * Test Data: (624, 150, 150, 1)
        * Test Labels: (624,)

3.  **Chest X-Ray Pneumonia COVID19 Tuberculosis**: A comprehensive dataset that also includes pneumonia and normal images.
    * **Source:** `/kaggle/input/chest-xray-pneumoniacovid19tuberculosis/`
    * **Shapes after loading and preprocessing (x_train2, y_train2, x_test2, y_test2):**
        * Training Data: (5216, 150, 150, 1)
        * Training Labels: (5216,)
        * Test Data: (624, 150, 150, 1)
        * Test Labels: (624,)

The `COVID19` and `TUBERCULOSIS` classes were intentionally excluded from all datasets to focus solely on distinguishing `PNEUMONIA` from `NORMAL` cases.

### Combined Dataset Statistics

The datasets were combined to create a larger and more diverse training and testing set. The `val` set from the second dataset (`chest-xray-pneumonia`) was used as the validation set during training.

| Dataset Split | PNEUMONIA (Class 0) | NORMAL (Class 1) | Total Samples | Shape (Data)      | Shape (Labels) |
| :------------ | :------------------ | :--------------- | :------------ | :---------------- | :------------- |
| **Training** | 11168               | 3948             | 15116         | (15116, 150, 150, 1) | (15116,)       |
| **Validation**| -                   | -                | 16            | (16, 150, 150, 1) | (16,)          |
| **Test** | 1635                | 785              | 2420          | (2420, 150, 150, 1)  | (2420,)        |

*Note: The exact validation set size is small (16 images) as provided by one of the source datasets. Data augmentation helps to mitigate this.*
## Methodology

The project follows a standard deep learning pipeline, with each step carefully designed to optimize model performance and reliability:

1.  **Data Loading and Preprocessing**:
    * **Action:** Images are loaded in grayscale, resized to 150x150 pixels, and normalized to a range of 0-1.
    * **Justification:**
        * **Grayscale:** Chest X-rays are typically grayscale images, so processing them as such reduces computational complexity without losing relevant diagnostic information.
        * **Resizing (150x150):** Standardizing image dimensions is crucial for feeding them into a neural network. 150x150 is chosen as a balance between retaining sufficient image detail and computational efficiency.
        * **Normalization (0-1):** Scaling pixel values from 0-255 to 0-1 helps the neural network converge faster and perform better. It prevents larger pixel values from dominating the learning process and aids in stable gradient computation.

2.  **Dataset Combination**:
    * **Action:** Multiple chest X-ray datasets are combined to increase the training and testing data size.
    * **Justification:**
        * **Increased Data Volume:** Deep learning models, especially CNNs, perform significantly better with more data. Combining datasets provides a larger and more diverse set of examples, which helps the model learn more robust features and generalize well to unseen data.
        * **Reduced Bias:** Different datasets might have slight variations in image acquisition or patient demographics. Combining them can help reduce potential biases inherent in a single dataset.

3.  **Data Augmentation**:
    * **Action:** `ImageDataGenerator` is used to apply various transformations to the training images, including rotation, zoom, width/height shifts, and horizontal flips.
    * **Justification:**
        * **Preventing Overfitting:** Data augmentation artificially expands the training dataset by creating modified versions of existing images. This exposes the model to a wider variety of data, making it less likely to memorize the training examples and improving its ability to generalize to new, slightly varied inputs.
        * **Improving Robustness:** These transformations simulate minor variations that might occur in real-world X-ray captures (e.g., slight patient positioning differences), making the model more robust to such variations.

4.  **Model Building**:
    * **Action:** A custom Convolutional Neural Network (CNN) architecture is designed.
    * **Justification:**
        * **Feature Learning:** CNNs are inherently well-suited for image classification tasks because they can automatically learn hierarchical features (e.g., edges, textures, shapes) directly from the raw pixel data, eliminating the need for manual feature engineering.
        * **Task Specificity:** A custom architecture allows for tailoring the network's depth, width, and layer configurations to the specific complexity and characteristics of chest X-ray images for pneumonia detection.

5.  **Model Training**:
    * **Action:** The CNN model is trained using the augmented training data and evaluated on the validation set. `ReduceLROnPlateau` callback is used to adjust the learning rate during training.
    * **Justification:**
        * **Learning from Data:** This is the core step where the model learns the patterns and relationships within the training data by iteratively adjusting its internal weights to minimize the loss function.
        * **Validation for Generalization:** Evaluating on a separate validation set (`x_val, y_val`) during training provides an unbiased estimate of the model's performance on unseen data and helps in detecting overfitting early.
        * **Adaptive Learning Rate (`ReduceLROnPlateau`):** Dynamically reducing the learning rate when validation accuracy plateaus helps the model escape local minima and fine-tune its weights more precisely, leading to better convergence and potentially higher accuracy.

6.  **Model Evaluation**:
    * **Action:** The trained model's performance is assessed using various metrics (Loss, Accuracy, Classification Report, Confusion Matrix, ROC Curve, Precision-Recall Curve, Class Distribution Comparison, Prediction Probability Distribution) on the combined test set.
    * **Justification:**
        * **Comprehensive Performance Assessment:** A single metric like accuracy can be misleading, especially with imbalanced datasets. Using a variety of metrics (precision, recall, F1-score, AUC) provides a more holistic and nuanced understanding of the model's strengths and weaknesses for both positive and negative classes.
        * **Unbiased Assessment:** The test set is completely separate from the training and validation sets, providing an unbiased measure of the model's true generalization capability on completely new data.

7.  **Model Saving**:
    * **Action:** The trained model is saved for future use.
    * **Justification:**
        * **Reproducibility and Deployment:** Saving the trained model allows for its easy reuse without retraining, which is essential for deployment, sharing, or continuing development. It ensures that the exact trained state of the model can be reloaded and applied to new data.
        * **Time Efficiency:** Retraining a deep learning model from scratch can be very time-consuming and computationally expensive. Saving the model avoids this overhead.
## Model Architecture

The CNN model architecture is sequential and consists of multiple convolutional layers, batch normalization, max-pooling layers, and dense layers. Dropouts are extensively used to prevent overfitting.

| Layer Type                 | Output Shape             | Parameters | Description                                                                   |
| :------------------------- | :----------------------- | :--------- | :---------------------------------------------------------------------------- |
| `Conv2D` (32 filters)      | (None, 150, 150, 32)     | 320        | Learns initial features. `(3,3)` kernel, `same` padding, `relu` activation. |
| `BatchNormalization`       | (None, 150, 150, 32)     | 128        | Normalizes activations.                                                       |
| `MaxPool2D`                | (None, 75, 75, 32)       | 0          | Downsamples feature maps. `(2,2)` pool size, `strides=2`.                     |
| `Conv2D` (64 filters)      | (None, 75, 75, 64)       | 18,496     | Further feature extraction.                                                   |
| `Dropout` (0.1)            | (None, 75, 75, 64)       | 0          | Prevents overfitting.                                                         |
| `BatchNormalization`       | (None, 75, 75, 64)       | 256        |                                                                               |
| `MaxPool2D`                | (None, 38, 38, 64)       | 0          |                                                                               |
| `Conv2D` (64 filters)      | (None, 38, 38, 64)       | 36,928     |                                                                               |
| `BatchNormalization`       | (None, 38, 38, 64)       | 256        |                                                                               |
| `MaxPool2D`                | (None, 19, 19, 64)       | 0          |                                                                               |
| `Conv2D` (64 filters)      | (None, 19, 19, 64)       | 36,928     |                                                                               |
| `BatchNormalization`       | (None, 19, 19, 64)       | 256        |                                                                               |
| `MaxPool2D`                | (None, 10, 10, 64)       | 0          |                                                                               |
| `Conv2D` (128 filters)     | (None, 10, 10, 128)      | 73,856     |                                                                               |
| `Dropout` (0.2)            | (None, 10, 10, 128)      | 0          |                                                                               |
| `BatchNormalization`       | (None, 10, 10, 128)      | 512        |                                                                               |
| `MaxPool2D`                | (None, 5, 5, 128)        | 0          |                                                                               |
| `Conv2D` (256 filters)     | (None, 5, 5, 256)        | 295,168    |                                                                               |
| `Dropout` (0.2)            | (None, 5, 5, 256)        | 0          |                                                                               |
| `BatchNormalization`       | (None, 5, 5, 256)        | 1,024      |                                                                               |
| `MaxPool2D`                | (None, 3, 3, 256)        | 0          | Final downsampling before flattening.                                         |
| `Flatten`                  | (None, 2304)             | 0          | Flattens the feature maps for the dense layers.                               |
| `Dense` (128 units)        | (None, 128)              | 295,040    | Fully connected layer.                                                        |
| `Dropout` (0.2)            | (None, 128)              | 0          |                                                                               |
| `Dense` (128 units)        | (None, 128)              | 16,512     |                                                                               |
| `Dropout` (0.2)            | (None, 128)              | 0          |                                                                               |
| `Dense` (128 units)        | (None, 128)              | 16,512     |                                                                               |
| `Dropout` (0.3)            | (None, 128)              | 0          |                                                                               |
| `Dense` (128 units)        | (None, 128)              | 16,512     |                                                                               |
| `Dropout` (0.3)            | (None, 128)              | 0          |                                                                               |
| `Dense` (1 unit)           | (None, 1)                | 129        | Output layer with `sigmoid` activation for binary classification.             |
| **Total Parameters** |                          | **808,833**|                                                                               |
| **Trainable Parameters** |                          | **807,617**|                                                                               |
| **Non-trainable Parameters** |                          | **1,216** | (from BatchNormalization layers)                                              |

## Training Details

* **Optimizer:** `rmsprop`
* **Loss Function:** `binary_crossentropy`
* **Metrics:** `accuracy`
* **Epochs:** 15
* **Batch Size:** 64
* **Callbacks:** `ReduceLROnPlateau`
    * Monitors `val_accuracy`
    * Patience: 2 epochs (if validation accuracy doesn't improve for 2 consecutive epochs, learning rate is reduced)
    * Factor: 0.3 (new learning rate = old learning rate * 0.3)
    * Minimum Learning Rate: 0.00000001

## Evaluation Metrics

The model's performance was evaluated using the following metrics on the combined test set:

* **Loss**
* **Accuracy**
* **Classification Report** (Precision, Recall, F1-score for each class)
* **Confusion Matrix**
* **ROC Curve and AUC Score**
* **Precision-Recall Curve and Average Precision Score**
* **Class Distribution Comparison** (Actual vs. Predicted)
* **Prediction Probability Distribution**

### Test Set Performance

| Metric              | Value      |
| :------------------ | :--------- |
| **Test Loss** | 0.1678     |
| **Test Accuracy** | 94.01%     |

### Classification Report

| Class                 | Precision | Recall | F1-Score | Support |
| :-------------------- | :-------- | :----- | :------- | :------ |
| **Pneumonia (Class 0)** | 0.96      | 0.95   | 0.96     | 1635    |
| **Normal (Class 1)** | 0.89      | 0.93   | 0.91     | 785     |
| **Accuracy** |           |        | 0.94     | 2420    |
| **Macro Avg** | 0.93      | 0.94   | 0.93     | 2420    |
| **Weighted Avg** | 0.94      | 0.94   | 0.94     | 2420    |

### Visualizations

The following plots are generated and saved in the `model_evaluation_graphs` directory to provide a visual understanding of the model's performance:

* `training_history.png`: Plots of training and validation accuracy and loss over epochs.
* ![training_history](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/training_history.png)

* `confusion_matrix.png`: A heatmap showing the confusion matrix.
* * ![confusion_matrix](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/confusion_matrix.png)

* `roc_curve.png`: Receiver Operating Characteristic curve.
* * * ![roc_curve](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/roc_curve.png)

* `precision_recall_curve.png`: Precision-Recall curve.
* * * * ![precision_recall_curve](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/precision_recall_curve.png)

* `class_distribution.png`: Bar plots comparing actual and predicted class distributions.
* * * * * ![class_distribution](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/class_distribution.png)

* `probability_distribution.png`: Histogram of prediction probabilities.
* * * * * * ![probability_distribution](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy/blob/main/probability_distribution.png)


## Results

The model achieved a test accuracy of approximately 94.01%. The classification report indicates strong performance across both classes, with high precision, recall, and F1-scores for identifying both Pneumonia and Normal cases. The ROC AUC and Average Precision scores (visible in the generated plots) further confirm the model's strong discriminatory power.
Usage

This project provides a pre-trained CNN model for pneumonia detection and a FastAPI application to serve predictions. You can use it out-of-the-box or modify and retrain the model.

Running the Application Locally

To run the pneumonia detection application on your local machine:

Clone the Repository:

Start by cloning this GitHub repository to your local machine:

Bash

git clone https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy.git
cd Pneumonia-Detection-via-CNN-94-Test-Accuracy
Set Up Your Environment:
Ensure you have Python installed (Python 3.8+ recommended). Then, install the necessary libraries. You can find all required dependencies in the requirements.txt file.

Bash

pip install -r requirements.txt

Model Availability:

The pre-trained model, pneumonia_detection_model.h5, is already included in the repository. The FastAPI application (app.py) is configured to load this model from the root directory.

Run the FastAPI Application:

Navigate to the root directory of the cloned repository in your terminal and start the FastAPI application using Uvicorn:

Bash

uvicorn app:app --reload

You should see output indicating the server is running, typically at http://127.0.0.1:8000.

Access the Web Interface:

Open your web browser and go to http://127.0.0.1:8000. You'll find a user-friendly interface where you can upload chest X-ray images and get instant predictions.


Retraining or Modifying the Model

If you wish to improve the model's performance, experiment with different architectures, or train on new data, you can do so by running the original Kaggle notebook and saving a new version of the model.

Access the Kaggle Notebook:

Go to the Kaggle notebook where this model was trained: Pneumonia Detection via CNN - 94% Test Accuracy.

Make Changes:

You can modify the notebook's code to:

Adjust the CNN architecture (e.g., add/remove layers, change filter counts).

Tweak hyperparameters (e.g., learning rate, batch size, number of epochs).

Experiment with different data augmentation techniques.

Incorporate additional datasets (if available and compatible).

Train and Save the New Model:

After making your desired changes, re-run the notebook to train the model. Ensure that the final step saves the trained model with the filename pneumonia_detection_model.h5. The relevant line in the notebook would be something like:

Python

model.save('pneumonia_detection_model.h5')

This will save the new version of your model as an output of the Kaggle notebook.

Download and Replace:

Once the notebook run is complete and the model is saved, download the pneumonia_detection_model.h5 file from your Kaggle notebook's output. Then, replace the existing pneumonia_detection_model.h5 file in your local GitHub repository with this newly trained version.


Restart the FastAPI Application:

If your FastAPI application is running, stop it (Ctrl+C in the terminal) and restart it using uvicorn app:app --reload. The application will automatically load your new model, reflecting any performance improvements or changes you've made.
