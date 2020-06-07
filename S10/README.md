
## EVA4 - Assignment 10

### Assignment: 

- Pick your last code
- Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
- Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.)
  - Move LR Finder code to your modules
  - Implement LR Finder (for SGD, not for ADAM)
  - Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
- Find best LR to train your model
- Use SDG with Momentum
- Train for 50 Epochs.
- Show Training and Test Accuracy curves
- Target 88% Accuracy.
- Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
- Submit


### Submission:

#### Parameters:
- Loss Function: Cross Entropy Loss
- L1 decay: 1e-6
- L2 decay: 1e-3
- Optimizer: SGD
- Initial Learning Rate: 0.0001
- Optimizer: SGD with momentum=0.9
- LR Finder used to update LR
- ReduceLROnPlateau used
- Model: Resnet18
- Model Parameters: default parameters with dropout=0.15 added
- Epochs: 50
- Gradcam on 25 misclassified images implemented

#### Image Augmentation Techniques - Albumentations
- HorizontalFlip
- Normalize
- Rotate
- HueSaturationValue
- Cutout

#### Results
- Target accuracy of 88% achieved
- Heighest Accuracy achieved: 91.56% (epoch:36)
- Observations:
  - Consistently above 90% accuracy from epoch 22 onwards.

#### Train vs Test Loss & Accuracy
<img src="https://github.com/aswa09/EVA-4/blob/master/S10/results/acc_vs_loss.jpg">

#### Sample Gradcam results for misclassified images:

##### Actual: Cat, Predicted: Dog
<img src="https://github.com/aswa09/EVA-4/blob/master/S10/results/gradcam/gradcam_incorrect_0_dog.png">
