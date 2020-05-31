## EVA4 - Assignment 9

### Assignment: 

- Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
- Please make sure that your test_transforms are simple and only using ToTensor and Normalize
- Implement GradCam function as a module. 
- Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
- Target Accuracy is 87%

### Submission:

#### Parameters:
- Loss Function: Cross Entropy Loss
- L1 decay: 1e-6
- L2 decay: 1e-3
- Optimizer: SGD
- Learning Rate: 0.01
- Model: Resnet18
- Model Parameters: default parameters with dropout(0.1) added
- Epochs: 25

#### Image Augmentation Techniques - Albumentations
- HorizontalFlip
- Normalize
- Rotate
- HueSaturationValue
- Cutout

#### Results
- Heighest Accuracy achieved: 88.64% (epoch:23)
- Observations:
  - Consistently above 80% accuracy from epoch 7 onwards.
  - Epoch 12 onwards have 85%+ accuracy (excluding epoch 14 which was close to 85% at 84.44%)

#### Gradcam results for misclassified images:

##### Actual: Cat, Predicted: Dog
<img src="https://github.com/aswa09/EVA-4/blob/master/S9/gradcam_incorrect_0_dog.png">

##### Actual: Plane, Predicted: Ship
<img src="https://github.com/aswa09/EVA-4/blob/master/S9/gradcam_incorrect_1_ship.png">

#### Test Loss & Test Accuracy
<img src="https://github.com/aswa09/EVA-4/blob/master/S9/acc_vs_loss.jpg">
