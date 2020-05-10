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

1. RF is greater then 44
2. convblock3 - Depthwise Separable Convolution
3. convblock5 - Dilated Convolution
4. Added GAP
5. Max Accuracy: 83.19% on EPOCH 18
6. Total Params: 141,376
