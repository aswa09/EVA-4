# Session 14 - RCNN FAMILY

Dataset Link: [Download from here](https://drive.google.com/file/d/1CFW9CqsZ3JkpOyC-tR_7Qttq57aXXdoF/view?usp=sharing)

## Directory Structure
- **bg**: Background images (100)
- **fg**: Foreground images (100)
- **fg_mask**: Mask of foreground images (100)
- **fg_bg**: 
    - 100 zip files divided based on bg
    - each bg zip contains 4000 images
    - Images where foregrounds are overlayed on top of backgrounds
- **fg_bg_mask**:
    - 100 zip files divided based on bg
    - each bg zip contains 4000 images
    -Mask of overlayed foreground-background images
- **fg_bg_depth**:
    - 100 zip files divided based on bg
    - each bg zip contains 4000 images
    - Depth maps of overlayed foreground-background images

- **labels.txt**: text file containing location of foreground in each background

## Sample Images:

**bg**
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/bg_grid.png" width="1000" />
</p>

**fg**
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/fg_grid.png" width="500" />
</p>

**fg_mask**
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/fg_mask_grid.png" width="500" />
</p>

**fg_bg** 
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/fg_bg_grid.png" width="1000" />
</p>

**fg_bg_mask**
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/fg_bg_mask_grid.png" width="1000" />
</p>

**fg_bg_depth**
<p style="text-align:center;">
    <img src="https://github.com/aswa09/EVA-4/blob/master/S14/results/fg_bg_depth_grid.png" width="1000" />
</p>

## Content Description

### bg
- **background images**.
- **Image Size**: 224x224x3
- **Number of Images**: 100
- **Naming Convention**: `bg001.jpeg`, `bg002.jpeg`, ..., `bg100.jpeg`

### fg
- **foreground images**
- **Image Height**: 108 (width will depend upon the aspect ratio of each image)
- **Number of Images**: 100
- **Naming Convention**: `fg001.png`, `fg002.png`, ..., `fg100.png`

### fg_mask
- **foreground masks**
- **Image Height**: 108
- **Number of Images**: 100
- **Naming Convention**: `fg001_mask.png`, `fg002_mask.png`, ..., `fg100_mask.png`

### bg_fg
- **background-foreground images**
- **Image Size**: 224x224x3
- **Number of Images**: 400,000
- **Naming Convention**: `bg001_fg001_01.jpeg`, `bg001_fg001_02.jpeg`, ..., `bg100_fg100_40.jpeg`

### bg_fg_mask
- **background-foreground masks**
- **Image Size**: 224x224x1
- **Number of Images**: 400,000
- **Naming Convention**: `bg001_fg001_01_mask.jpeg`, `bg001_fg001_02_mask.jpeg`, ..., `bg100_fg100_40_mask.jpeg`

### bg_fg_depth_map
-**background-foreground depth maps**
- **Image Size**: 224x224x1
- **Number of Images**: 400,000
- **Naming Convention**: `bg001_fg001_01_depth_map.jpeg`, `bg001_fg001_02_depth_map.jpeg`, ..., `bg100_fg100_40_depth_map.jpeg`

### labels.txt
- Each line in the file contains 5 entries separated by `\t` where entries are ordered as `background-foreground`, `bbox top left x-coordinate`, `bbox top left y-coordinate`, `bbox width`, `bbox-height`.

## Statistics

[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aswa09/EVA-4/blob/master/S14/EVA4_S15A_data_statistics.ipynb)

- Dataset Size: 
- Number of Images: 1,200,100
- Image Types and their Statistics
  - **Backgrounds**
    - Image Size: 224x224x3
    - Number of Images: 100
    - Mean: (0.40086, 0.46599, 0.53281)
    - Standard Deviation: (0.25451, 0.24249, 0.23615)
  - **Background-Foregrounds**
    - Image Size: 224x224x3
    - Number of Images: 400,000
    - Mean: (0.41221, 0.47368, 0.53431)
    - Standard Deviation: (0.25699, 0.24577, 0.24217)
  - **Background-Foreground Masks**
    - Image Size: 224x224x1
    - Number of Images: 400,000
    - Mean: 0.05207
    - Standard Deviation: 0.21686
  - **Background-Foreground Depth Maps**
    - Image Size: 224x224x1
    - Number of Images: 400,000
    - Mean: 0.2981
    - Standard Deviation: 0.11561

## Dataset Creation

[![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aswa09/EVA-4/blob/master/S14/EVA4_S15A_data_generation.ipynb)

The dataset was created as follows

- [Download background and foreground images](#download-background-and-foreground-images)
- [Background removal from foregrounds](#background-removal-from-foregrounds)
- [Mask creation for foregrounds](#mask-creation-for-foregrounds)
- [Creating mask](#creating-mask)
- [Overlaying foregrounds on backgrounds](#overlaying-foregrounds-on-backgrounds)
- [Dataset Creation](#creating-mask)
- [Depth map](#depth-map)
- [Modifications](#modifications)

### Download background and foreground images

- Create the directories `bg` and `fg` inside the **root directory**.
- bg:
    - Search and download **100 images** of backgrounds.
    - Crop these images to a square (`704x704`)**
    - Done in MS Paint.
    - Keep these images in the `bg` directory.
- fg:
    - Search and download **100 images** containing people.
    - Keep these images in the `fg` directory.

**all the images will be resized to a size of 224x224.

### Background removal from foregrounds

- All the downloaded foreground images should have a transparent background in order to overlay them on top of background images.
- For removing backgrounds, **GIMP - GNU Image Manipulation Program** can be used
- Steps for removing background using GIMP has been shown below:
- Images are exported as PNG because to retain transparency.

### Mask creation for foregrounds

- There is an alpha channel in the in foreground images which specifies the opacity for a color. After adding transparent backgrounds to the images in _GIMP_, the alpha parameter ranges from 0 (fully transparent) to 255 (fully opaque).
- The alpha channel in foreground images has pixel value set to 0 wherever transparency is present.
- After adding transparency to images in _GIMP_, the background color of the image is set to white (i.e. pixels values in RGB channel are equal to 255) which is hidden with the help of the alpha channel.

#### Creating mask

- The pixels in the foreground image are set to 255 (white) where the object is present and rest of the pixels (background) are set to 0 (black).
- The non-zero values in the alpha channel are set to 255, this ensures full opaqueness of the object mask in the image.
- This can be done in GIMP

### Overlaying foregrounds on backgrounds

Foregrounds are overlayed on backgrounds to create background-foreground images. A corresponding mask of background-foreground image will also be created.

#### Dataset Creation
Now for each background-foreground pair:
1. Find a random location (x,y) on the background image. Make sure that x ranges between `[0, background_height-foreground_height]` and y ranges between `[0, background_width-foreground_width]`. This ensures that foreground is always completely inside the background.
1. Place the foreground image on top of the background image with (x,y) as the top left corner. This will be the background-foreground image.
1. Calculate the bounding box values for this image
    - Current image size is 704x704 and the final image size will be 224x224, so the bounding box values will be calculated with respect to the final image size
    - Calculate the scale ratio = 224 / 704
    - Multiply the x, y, foreground width and height values with the scale ratio to obtain the bounding box values
1. Place the mask of the foreground on a black image that has same shape as that of the background at (x,y) as the top left corner. This will be the mask for the background-foreground image.
1. Since the mask is in grayscale, the number of channels in these images is reduced to 1 to save storage space.

- Repeat the steps for 20 different locations.
- Flip the foreground and its mask horizontally and again repeat the steps for 20 different locations.
- Store the fg location of all these images to a file.

#### Images created in this step

- Number of foreground images = 100
- Number of background images = 100
- Overlaying foreground on 20 different locations: 100x100x20 = 200,000
- Overlaying horizontally flipped foreground on 20 different locations: 100x100x20 = 200,000
- Total overlayed images = 400,000

Each of *background-foreground*, *background-foreground mask* and *background-foreground depth map* will have 400,000 images.

Thus, total number of images in the dataset is **1,200,000**.

### Depth map

To create the monocular depth estimation map of the background-foreground images, we use pretrained DenseNet-201. Implementation for the model inference was referenced from [this](https://github.com/ialhashim/DenseDepth) repository.

### Modifications:
As the basic file structure was different from the original, the code for testing, etc has been modified to work on zip files directly and store the results in zip files of the same structure.
