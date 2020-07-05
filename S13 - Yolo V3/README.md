# Session 13 - YOLO 2 & 3

### Part 1 - YOLO v3 with OpenCV

  Detecting objects in an image where there is a person and an object from the COCO classes present in the image.  

  #### Result
  <p style='text-align:center;'>
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/results/Labelled.png" width="400"/>
  </p>

### Part 2 - YOLO v3 with PyTorch

  Object detection with Yolo v3 on a class that doesn't belong to COCO dataset.
  Object(s): Chess pieces
  Classes:king, queen, rook, bishop, knight and pawn

  ### Result

  Click on the video below to play
  [](https://youtu.be/QT1lcrZq3kY 'Chess Piece detection with Yolo V3 - GM Kasparov vs GM Karpov (1987)')

  ### Parameters and Hyperparameters

  - Number of images: 500
  - Batch size: 10
  - Epochs: 300

  ### Dataset Preparation

  For using the dataset, follow the instructions mentioned [here](YoloV3/data/customdata/README.md).
  To run the model on custom dataset, follow the steps below

  #### Train Data

  - Annotating the images
    - Clone the annotation tool from this [link](https://github.com/miki998/YoloV3_Annotation_Tool).
    - Follow the steps mentioned in the README of the tool specified above.
    - Annotate atleast 500 images with the tool.
  - Creating dataset directory
    - Place the annotated images [here](YoloV3/data/images).
    - Place the labels [here](YoloV3/data/labels).

  #### Test Data

  - Download a short-duration video containing the class used during training.
  - Extract frames from the video into the [images folder](YoloV3/data/customdata/images) directory  
    `!ffmpeg -i kasporov_vs_karpov.mp4 -vf fps=30 image-%04d.jpg`

  ### Downloading Pre-Trained Weights

  - Download the file named 'yolov3-spp-ultralytics.pt' from this [link](https://drive.google.com/file/d/1x51LwxMfFk3W1j2qTJmG68p7cBzo4NNs/view?usp=sharing) and place it in [this](YoloV3/weights) directory.

  ### Inference on a Video

  - Combine the images from the [output](YoloV3/output) directory to form a video  
    `!ffmpeg -framerate 30 -i output/image-%04d.jpg -r 30 -y kasporov_vs_karpov_od.mp4`

  ### Results

  - After running the algorithm for 300 epochs, some sample results:
  <p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/YoloV3/output/image-0003.jpg" width="400" />
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/YoloV3/output/image-3373.jpg" width="400" />
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/YoloV3/output/image-3635.jpg" width="400" />
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/YoloV3/output/image-3790.jpg" width="400" />
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/YoloV3/output/image-3876.jpg" width="400" />
  </p>

  - Training collab:
  <p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/results/train_batch0.png" width="800" />
  </p>

  - Testing collab:
  <p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/results/test_batch0.png" width="800" />
  </p>

  - Metrics:
  <p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S13%20-%20Yolo%20V3/results/results.png" width="800" />
  </p>
