# Advanced Convolutions

## Assignment

1. Change the code such that it uses GPU
2. Change the architecture to C1C2C3C40 (3 MPs)
3. Total RF must be more than 44
4. One of the layers must use Depthwise Separable Convolution
5. One of the layers must use Dilated Convolution
6. Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. Achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M

## Results

1. RF > 44
2. Epochs: 25
3. accuracy > 80% (consistently from epoch 15 onwards)
4. Added GAP
5. Added Depthwise Separable Convolution
6. Added Dilated Convolution
