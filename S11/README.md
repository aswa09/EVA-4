

# Session 11 - Super Convergence

Assignment:

- Write a code that draws this curve (without the arrows). - Triangular wave
- Write a code which uses this new ResNet Architecture for Cifar10:
  1.  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  2.  Layer1 -
      1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      2.  R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
      3.  Add(X, R1)
  3.  Layer 2 -
      1.  Conv 3x3 [256k]
      2.  MaxPooling2D
      3.  BN
      4.  ReLU
  4.  Layer 3 -
      1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      2.  R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      3.  Add(X, R2)
  5.  MaxPooling with Kernel Size 4
  6.  FC Layer
  7.  SoftMaxion
- Uses One Cycle Policy such that:
  1. Total Epochs = 24
  2. Max at Epoch = 5
  3. LRMIN = FIND
  4. LRMAX = FIND
  5. NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Target Accuracy: 90%. 
- The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you'd get. 

###  Parameters and Hyperparameters
- Loss Function: Negative Log Likelihood
- Optimizer: SGD
- Scheduler: OneCycleLR
- Batch Size: 512
- Epochs: 24
- L1 decay: 2e-6
- L2 decay: 6e-4

### Image Augmentation Techniques
- PadIfNeeded
- RandomCrop
- HorizontalFlip
- Cutout
- HueSaturationValue
- Rotate
- Normalize

### Learning Rate Parameters
OneCycleLR:
- max_lr: computed
- div_factor: 10
- final_div_factor: 1
- anneal_strategy: linear
- pct: 5/24

### Results
Achieved  an accuracy of **89%** 

#### Train vs Test Accuracy
<img src="https://github.com/aswa09/EVA-4/blob/master/S11/results/acc_trn_vs_tst.png">

#### LR Scheduler:
<img src="https://github.com/aswa09/EVA-4/blob/master/S11/results/LR_scheduler.png">

#### Triangular wave:
<img src="https://github.com/aswa09/EVA-4/blob/master/S11/results/triangular.png">
