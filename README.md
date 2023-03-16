# pytorch_medical_image_analysis

This repo aims to create a pytorch warehouse for (medical) image analysis consisting of classification, detection, segmentation (2D and 3D)

## Classification

Our goal for this part is to train a classifier that can predict whether a patient's X-Ray indicates the presence of pneumonia or not. To achieve this, we will be utilizing the RSNA Pneumonia Detection Challenge dataset, which can be found at the following link: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

The task is a typical binary classification problem with samples shown below (1 indicates positive while 0 indicates negative).

![mia_classification](https://user-images.githubusercontent.com/6441064/225615386-3b088e8b-cd5c-45eb-8b4f-0e4f622b74fb.png)

The code does not use handle imbalance issue for this dataset.  Cross entropy is used as the loss function and accuracy is used as a metric for evaluation.
The confusion matrix for the validation dataset is shown below

![confusionMatrix](https://user-images.githubusercontent.com/6441064/225621968-c15ca4af-5bce-4686-90a4-5e0d74f7ebd4.png)
