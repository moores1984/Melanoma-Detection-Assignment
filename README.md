# Project Name

**Project Name: CNN-Based Melanoma Detection**

This project aims to build a custom CNN model that accurately detects melanoma, a type of cancer responsible for 75% of skin cancer deaths. The model evaluates images to assist dermatologists in early detection, reducing the manual effort required in diagnosis.

Please note, Uploaded images in the Repo are from BEFORE ipynb file has been run so that anyone who downloads this, can run their own instance.


## Table of Contents
- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Conclusions](#conclusions)
    - [Summary:](#summary)
  - [Technologies Used](#technologies-used)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)


## General Information

 - Background: Building a CNN model to detect melanoma, one of the deadliest forms of skin cancer.
 - Business Problem: Reducing manual diagnosis effort and enabling early detection.
 - Dataset Used: 2357 images of malignant and benign oncological diseases from the ISIC, including various diseases like Melanoma, Nevus, Basal cell carcinoma, etc.

## Conclusions

 - Final Training Accuracy: The model learned the Training Data Effectively.
 - Final Validation Accuracy: The model performed well on unseen data suggesting good generalization.
 - Low Training and Validation Loss
 - Improved accuracy and robustness in the latest model
 - Considerable enhancement through balanced regularization, simplified architecture, and Augmentor usage.

### Summary:

 - The final model demonstrates strong performance in both training and validation.
 - Automated hyperparameter tuning and feature selection methods are recommended for further enhancements.
 - The third model significantly outperformed the previous two, illustrating the importance of a structured approach to tuning.


## Technologies Used

 - TensorFlow and Keras: The backbone of the entire project, TensorFlow's Keras API was used to design, compile, and train the CNN model.
 - Custom CNN Model Architecture:
   - Normalization layer: To rescale the pixel values.
   - Convolutional layers: Four blocks with Conv2D layers followed by MaxPooling. Filter sizes include 32, 64, 128, and 256, with kernel sizes ranging from 3x3 to 11x11.
   - Dropout layers: For regularization and to prevent overfitting. Drop rates used include 0.5 and 0.25.
   - Flattening layer: To transform the 2D matrix data to a vector before building the fully connected layers.
   - Dense layers: Three fully connected layers with 256, 128, and 64 neurons, respectively.
   - Output layer: Multi-class classification with softmax activation.
 - Optimization and Compilation:
   - Optimizer: Adam optimizer with legacy support.
   - Loss function: Sparse categorical cross-entropy.
   - Metrics: Accuracy.
 - Training Details:
   - Augmentor Library: Used for data augmentation to rectify class imbalances.
   - Data Preprocessing: Utilized tf.keras.preprocessing.image_dataset_from_directory to load the data from directories and create training and validation datasets.
   - Batch size: 32
   - Image size: 180x180 pixels.
   - Epochs: 50 for the final training.
   - Callbacks: Custom callbacks (e.g., learn_control) can be used to customize the training process.
 - Other Tools:
   - Custom CNN Model
   - Image Preprocessing Libraries

## Acknowledgements
The dataset is formed from the International Skin Imaging Collaboration (ISIC).
Project guidelines and requirements.
This project was inspired by the need for automated early detection of melanoma.


## Contact
Created by [@moores1984]
