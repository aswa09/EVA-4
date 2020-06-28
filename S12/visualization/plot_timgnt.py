import random
import matplotlib.pyplot as plt
from utils.utils import unnormalize

def visualise_tinyimgnt(dataset,args):
  # Fetch data
  classes = dataset.classes

  # Set number of images to display
  num_images = 4

  # Display images with labels
  fig, axs = plt.subplots(1, 4, figsize=(8, 8))
  fig.tight_layout()

  for i in range(num_images):
      idx = random.randint(0, len(dataset.val_data))
      axs[i].axis('off')
      axs[i].set_title(f'Label: {classes[dataset.val_data[idx][1]].split(",")[0]}')
      axs[i].imshow(unnormalize(dataset.val_data[idx][0],args.mean, args.std, transpose=True))