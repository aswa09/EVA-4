# CODE 1: Basic model and setup
[Code link](https://github.com/aswa09/EVA-4/blob/master/S5/EVA4_S5_F1.ipynb)
## Target: basic model with setup	
1. get train and test data
2. set transforms
3. set data loaders
4. setup base code
5. no fancy stuff
5. set training & test loop
## Result:
Parameters: 83,898	
Best Train Accuracy: 99.76%	
Best Test Accuracy: 99.19% (EPOCH 13)
## Analysis: 
1. The best training accuracy is significantly more than best testing accuracy 
2. The number of parameters is too much
3. Overfitted

# CODE 2: Reducing number of parameters
[Code link](https://github.com/aswa09/EVA-4/blob/master/S5/EVA4_S5_F2.ipynb)
## Target:
1. Changed the number of kernels and layers to reduced the number of parameters
## Result:
Parameters: 9,466
Best Train Accuracy: 99.48%
Best Test Accuracy: 99.14% (EPOCH 13)
## Analysis:
1. Number of parameters successfully reduced
2. The Training accuracy & test accuracy has decreased due to decrease in parameters 
3. Margin between Train and Test accuracies has reduced, has potential to improve

# CODE 3: Batchnorm
[Code link](https://github.com/aswa09/EVA-4/blob/master/S5/EVA4_S5_F3.ipynb)
## Target: 	
1. Batchnorm included after every convolution to improve performance
## Result:
Parameters: 9,622	
Best Train Accuracy: 99.87%
Best Test Accuracy: 99.29% (EPOCH 10)
## Analysis:
1. There is more consistency in the test accuracy from the 10th epoch
2. we can try making model more efficient by regularizing the data

# CODE 4: Regularization(dropout)
[Code link](https://github.com/aswa09/EVA-4/blob/master/S5/EVA4_S5_F4.ipynb)
## Target:  
1. Added dropout as regularization to increase both train and test accuracy
## Result:
Parameters: 9,622
Best Train Accuracy: 99.27%
Best Test Accuracy: 99.43% (EPOCH 14)
## Analysis:
1. Margin between training and test accuracy has reduced drastically
2. under-fitting
3. There is even more consistency in the test accuracy from the 8th epoch (excluding epoch 13)

# CODE 5: Augmentation
[Code link](https://github.com/aswa09/EVA-4/blob/master/S5/EVA4_S5_F5.ipynb)
## Target:	
1. Random rotation of 5 degrees included in Training dataset to increase capability
## Result:
Parameters: 9,622
Best Train Accuracy: 99.12%
Best Test Accuracy: 99.40% (EPOCH 13)
## Analysis:
1. The model is under-fitting now, as we have made our train data harder.
2. The last 2 epochs close to target
