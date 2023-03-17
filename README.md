# pytorch_medical_image_analysis

This repo aims to create a pytorch warehouse for (medical) image analysis consisting of classification, detection, segmentation (2D and 3D)

## Classification

The goal for this part is to train a classifier that can predict whether a patient's X-Ray indicates the presence of pneumonia or not. To achieve this, we will be utilizing the RSNA Pneumonia Detection Challenge dataset, which can be found at the following link: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

The task is a typical binary classification problem with samples shown below. Here the input is the image and output is a label (1 indicates positive while 0 indicates negative).

![mia_classification](https://user-images.githubusercontent.com/6441064/225622764-92f8eff7-39bd-4bed-b33b-1711f043235c.png)

The code does not use handle imbalance issue for this dataset as for now. The resnet neural network is used for the algorithn. Cross entropy is used as the loss function and accuracy is used as a metric for evaluation.

The confusion matrix for the validation dataset is shown below

![mia_confusionMatrix](https://user-images.githubusercontent.com/6441064/225622827-099e1484-7e50-4f19-ba9f-0342a06bdbc5.png)

## Detection

The goal for this part is to predict a bounding box containg heart in the above X-Ray images.

The task can be formulated as a regression problem. Again the input is the image but the output is a 4d vector (xmin, ymin, xmax, ymax) which defines the location of the heart  as illustrated below. Here the Mean Square Error is used as the loss function.

![mia_detection](https://user-images.githubusercontent.com/6441064/225624864-b18069f5-cb77-499e-b10c-0337142580a1.png)

## 2D Segmentation

The goal for this part is to train a train a UNET structure model that segment atrium of a heart.  The input is the image and the output is an image with the same size indicating the object. 

![segmentation](https://user-images.githubusercontent.com/6441064/225810217-1e78d444-85b3-4e2c-9bfd-23d4235a0d1d.png)

