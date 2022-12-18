import numpy as np
import os
import torch
import torch.utils.data
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

DEBUG = False

class DTSegmentationDataset(torch.utils.data.Dataset):
    """
    Dataloader for the Duckietown dataset.
    Loads the images and the corresponding segmentation targets.
    """
    PATH_TO_ANNOTATIONS = "offline learning/data/annotations/"
    PATH_TO_IMAGES = "offline learning/data/frames/"
    CVAT_XML_FILENAME = "segmentation_annotation.xml"
    SEGM_LABELS = {
        'Background': {'id': 0, 'rgb_value': [0, 0, 0]}, # black
        'Ego Lane': {'id': 1, 'rgb_value': [102, 255, 102]}, # green
        'Opposite Lane': {'id': 2, 'rgb_value': [245, 147, 49]}, # orange
        'Obstacle': {'id': 3, 'rgb_value': [184, 61, 245]}, # purple
        'Road End': {'id': 4, 'rgb_value': [250, 50, 83]}, # red
    }
    
    def __init__(self, root):
        super(DTSegmentationDataset, self).__init__()
        self.root = root
        # load the annotation file and get the image names
        self.imgs = # TODO: get the image names from the annotation file
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "training/image_2"))))
        self.targets = list(sorted(os.listdir(os.path.join(root, "training/gt_image_2"))))
    
    def __getitem__(self, idx):
        if DEBUG:
            print(f"Image {self.imgs[idx]}, target {self.targets[idx]}")
        # load the image
        img_path = os.path.join(self.root, "training/image_2", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=SMALLEST_IMG_DIMS)])(img)
        
        # load the target
        target_path = os.path.join(self.root, "training/gt_image_2", self.targets[idx])
        target = np.array(Image.open(target_path).convert("RGB"))
        
        # labels are encoded as rgb values
        # Generate len(labels) channels with 1s where the label is present and 0s otherwise
        target_labels = np.zeros((len(SEG_LABELS_LIST), target.shape[0], target.shape[1])).astype(np.float32)
        
        # instances are encoded as different colors
        # Generate a label image with the same dimensions as the target image, i.e.
        # Target labels will have a shape of (h, w) but only one channel with each pixel labeled with label id
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            if DEBUG:
                print(f"Label {label['name']} has {np.sum(mask)} pixels. Assigning them to channel {label['id']}")
            target_labels[label['id'], mask] = 1.0
        
        target_labels = torch.from_numpy(target_labels.copy())        
        target_labels = transforms.CenterCrop(size=SMALLEST_IMG_DIMS)(target_labels)
        
        if DEBUG:
            print(f"Final image shape: {img.shape}, final target labels shape: {target_labels.shape}")
            print(f"Final image tensor type: {img.type()}, final target labels type: {target_labels.type()}")
        
        return img, target_labels
    
    def __len__(self):
        return len(self.imgs)


# ---------------------
# Randomly select a batch of images and masks from the dataset 
# and visualize them to check if the dataloader works correctly

if DEBUG:
    dataset = KITTIDataset("data_alt")
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=not DEBUG)

    # Get the first batch of images and masks
    images_batch, targets_batch = next(iter(data_loader))
    
    # Select the first image and mask from the batch
    image, target_labels = images_batch[0], targets_batch[0]
    
    # Convert the target labels to rgb to visualize them
    rgb_target = label_img_to_rgb(target_labels)
    # Visualize the image and target
    transforms.ToPILImage()(image).show()
    transforms.ToPILImage()(rgb_target).show()




# # Load the image from data_test
# image = Image.open(PATH_TO_IMAGES + image_name)
# image.show()

# # Create a draw tool to draw on the image
# drawer = ImageDraw.Draw(image)

# # Fill the polygon
# drawer.polygon(points, fill="red", outline="red")

# image.show()