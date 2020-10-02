

# Session 12 - Object Localization - YOLO

## Assignment:

  ### Assignment A:
  [Code link](https://github.com/aswa09/EVA-4/blob/master/S12/EVA4_S12_tinyimagenet.ipynb)
  
    - Download this TINY IMAGENET (Links to an external site.) dataset. 
    - Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
    - Submit Results. Of course, you are using your own package for everything. You can look at this (Links to an external site.) for reference. 

  ### Assignment B:
  [Code link](https://github.com/aswa09/EVA-4/blob/master/S12/EVA4_S12_dogs.ipynb)
  
    - Download 50 images of dogs. 
    - Use this (Links to an external site.) to annotate bounding boxes around the dogs.
    - Download JSON file. 
    - Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
    - Refer to this tutorial (Links to an external site.). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub.  

## Solution:

### Assignment A:

  ###  Parameters and Hyperparameters
    - Loss Function: Cross Entropy Loss
    - Optimizer: SGD
    - Scheduler: ReduceLROnPlateau
    - Batch Size: 128
    - Epochs: 50
    - momentum: 0.9
    - lr: 0.01

  ### Image Augmentation Techniques  
    - HorizontalFlip
    - VerticalFlip
    - Cutout
    - HueSaturationValue
    - Rotate
    - Normalize

  ### Results  
    - Achieved  an accuracy of ** 56.86% ** 

  #### Sample data:
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/tinyimagenet/timgnet_sample.png">
  
  #### Train vs Test Accuracy:
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/tinyimagenet/acc_trn_vs_tst.jpg">

  #### Gradcam for 3 misclassified images:
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/tinyimagenet/gradcam.png">
  
### Assignment B:

  ###  Json file description:'dogs_coco.json'
    - The annotation file is a JSON COCO format file, and with the below contents:

    - images:

      - id: Image id
      - width: Width of the image
      - height: Height of the image
      - filename: Image file name
      - license: License id for the image
      - date_captured: Date of capture of the image

    - annotations:

      - id: Annotation id
      - image_id: Id of the image the annotation is associated with
      - category_id: Id of the class the annotation belongs to
      - segmentation: (x, y) coordinates of the four corners of the bounding box
      - area: Area of the bounding box
      - bbox: (x, y) coordinate of the top-left corner and width and height of the bounding box
      - iscrowd: If the image has a crowd of objects denoted by this annotation

  ### Results
    - Best total numbers of clusters: 3 or 4 

  #### Height vs width of normalized bounding boxes:
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/width_vs_height.jpg">

  #### Number of Clusters(k) vs Mean IoU:
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/kmeans_iou.png">
  
  #### Cluster Plot for k=3
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/cluster_plot_k3.jpg">
  
  #### Cluster Plot for k=4
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/cluster_plot_k4.jpg">
  
  #### Anchor boxes for k=3
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/anchor_bbox_k3.png">
  
  #### Anchor boxes for k=4
  <img src="https://github.com/aswa09/EVA-4/blob/master/S12/results/anchor%20boxes/anchor_bbox_k4.png">
