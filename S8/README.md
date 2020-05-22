
## EVA4 - Assignment 8

### Assignment: 

- Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
- Extract the ResNet18 model from this repository and add it to your API/repo. 
- Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
- Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
- Once done finish S8-Assignment-Solution.

### Submission:

#### Parameters:

- Loss Function: Cross Entropy Loss
- L1 decay: 1e-6
- L2 decay: 1e-3
- Optimizer: SGD
- Learning Rate: 0.01
- Model: Resnet18
- Model Parameters: default parameters
- Epochs: 25

#### Image Augmentation Techniques
- Random Horizontal Flip: 0.5
- Normalizing

#### Results
- Heighest Accuracy achieved: 86.37% (epoch:25)
- Last accuracy achieved: 86.37% (epoch:25)
- Observations:
  - Consistently above 80% accuracy from epoch 6 onwards.
  - Epoch 19 onwards have 85%+ accuracy (excluding epoch 21 which was close to 85% at 84.73%)

#### Test Loss & Test Accuracy
<img src="https://github.com/aswa09/EVA-4/blob/master/S8/acc_vs_loss.jpg">
