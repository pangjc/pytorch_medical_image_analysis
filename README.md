# pytorch_medical_image_analysis

This repo aims to create a pytorch warehouse for (medical) image analysis consisting of classification, detection, segmentation (2D and 3D). The codes/methods should be easily applied to other types of images due to the fact that all images are matrices storing intensity values. There might be special python packages need to be installed to handle the specific MRI or CT image format (i.e. dicom). 

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

The goal for this part is to train a 2D UNET model that segment atrium of a heart.  The input is the image and the output is an image with the same size indicating the object. 

![Screenshot 2023-03-17 at 5 22 53 PM](https://user-images.githubusercontent.com/6441064/226056148-06a63040-1c0f-47fe-99ed-4a8ded410a4c.png)

## 3D Segmentation

Instead of performing segmentation for each 2D slice as above, the goal for this part is to train a 3D UNET model to segment atrium of a heart for the same 3D MR dataset as in the Section above.


