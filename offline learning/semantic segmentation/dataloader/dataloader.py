import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
from torchvision import transforms

from cvat_preprocessor import CVATPreprocessor

DEBUG = False

class DTSegmentationDataset(torch.utils.data.Dataset):
    """
    Dataloader for the Duckietown dataset.
    Loads the images and the corresponding segmentation targets.
    """
    PATH_TO_ANNOTATIONS = "offline learning/semantic segmentation/data/annotations/"
    PATH_TO_IMAGES = "offline learning/semantic segmentation/data/frames/"
    CVAT_XML_FILENAME = "segmentation_annotation.xml"
    SEGM_LABELS = {
        'Background': {'id': 0, 'rgb_value': [0, 0, 0]}, # black
        'Ego Lane': {'id': 1, 'rgb_value': [102, 255, 102]}, # green
        'Opposite Lane': {'id': 2, 'rgb_value': [245, 147, 49]}, # orange
        'Obstacle': {'id': 3, 'rgb_value': [184, 61, 245]}, # purple
        'Road End': {'id': 4, 'rgb_value': [250, 50, 83]}, # red
    }
    
    def __init__(self):
        super(DTSegmentationDataset, self).__init__()
        # Store the list of all image names
        self.imgs = CVATPreprocessor.get_all_image_names(self.PATH_TO_ANNOTATIONS + self.CVAT_XML_FILENAME)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        if DEBUG:
            print(f"Fetching image {image_name}")
        # load the image
        img = Image.open(self.PATH_TO_IMAGES + image_name).convert("RGB")
        
        # load the associated segmentation mask (list of polygons)
        all_polygons = CVATPreprocessor.get_all_image_polygons(image_name, self.PATH_TO_ANNOTATIONS + self.CVAT_XML_FILENAME)
        
        # Create a target image with the same spacial dimensions as the original image 
        # but a separate channel for each label
        target = np.zeros((len(self.SEGM_LABELS), img.size[1], img.size[0])).astype(np.float32)
        
        # Fill each channel with 1s where the corresponding label is present and 0s otherwise
        for label, polygons in all_polygons.items():
            # Create an empty bitmask for the current label and draw all label-associated polygons on it
            mask = Image.new('L', img.size, 0)
            drawer = ImageDraw.Draw(mask)
            for polygon in polygons:
                drawer.polygon(polygon, outline=255, fill=255)
            # Show the mask for extra debugging
            # mask.show()

            mask = np.array(mask) == 255
            if DEBUG:
                print(f"Label '{label}' has {np.sum(mask)} pixels. Assigning them to channel {self.SEGM_LABELS[label]['id']}")
            target[self.SEGM_LABELS[label]['id']][mask] = 1.0
        
        img = transforms.Compose([transforms.ToTensor()])(img)
        target = torch.from_numpy(target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def label_img_to_rgb(label_img):
        """
        Converts a label image (with one channel per label) to an RGB image.
        """
        rgb_img = np.zeros((label_img.shape[1], label_img.shape[2], 3), dtype=np.uint8)
        for label, label_info in DTSegmentationDataset.SEGM_LABELS.items():
            mask = label_img[label_info['id'], :, :] == 1.0
            rgb_img[mask] = label_info['rgb_value']
        return rgb_img


# ---------------------
# Randomly select a batch of images and masks from the dataset 
# and visualize them to check if the dataloader works correctly

if __name__ == "__main__":
    if DEBUG:
        dataset = DTSegmentationDataset()
        image, target = dataset[0]
        transforms.ToPILImage()(image).show()
        transforms.ToPILImage()(DTSegmentationDataset.label_img_to_rgb(target)).show()