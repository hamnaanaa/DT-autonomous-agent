import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
from torchvision import transforms

from . import cvat_preprocessor as cvat
# from cvat_preprocessor import CVATPreprocessor

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
        'Middle Lane': {'id': 1, 'rgb_value': [255, 255, 0]}, # yellow
        'Ego Lane': {'id': 2, 'rgb_value': [102, 255, 102]}, # green
        'Side Lane': {'id': 3, 'rgb_value': [255, 255, 255]}, # white
        'Opposite Lane': {'id': 4, 'rgb_value': [245, 147, 49]}, # orange
        'Intersection': {'id': 5, 'rgb_value': [50, 183, 250]}, # blue
        'Road End': {'id': 6, 'rgb_value': [250, 50, 83]}, # red
        'Obstacle': {'id': 7, 'rgb_value': [184, 61, 245]}, # purple
    }
    
    def __init__(self):
        super(DTSegmentationDataset, self).__init__()
        # Store the list of all image names
        self.imgs = cvat.CVATPreprocessor.get_all_image_names(self.PATH_TO_ANNOTATIONS + self.CVAT_XML_FILENAME)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        if DEBUG:
            print(f"Fetching image {image_name}")
        # load the image
        img = Image.open(self.PATH_TO_IMAGES + image_name).convert("RGB")
        
        # load the associated segmentation mask (list of polygons)
        all_polygons = cvat.CVATPreprocessor.get_all_image_polygons(image_name, self.PATH_TO_ANNOTATIONS + self.CVAT_XML_FILENAME)
        
        # Create a target image with the same spacial dimensions as the original image 
        # but a separate channel for each label
        target = np.zeros((480, 640)).astype(np.longlong)
        
        # Generate a random angle for rotation only once for both the image and the mask
        random_angle = np.random.randint(-10, 10)
        
        # Fill each channel with 1s where the corresponding label is present and 0s otherwise
        # Sort the labels by their corresponding IDs to ensure that the channels are filled in the correct order
        for label, polygons in sorted(all_polygons.items(), key=lambda x: self.SEGM_LABELS[x[0]]['id']):
            
            # Skip classes here if needed
            if label in ['Obstacle', 'Road End']:
                continue
            
            # Create an empty bitmask for the current label and draw all label-associated polygons on it
            mask = Image.new('L', img.size, 0)
            drawer = ImageDraw.Draw(mask)
            for polygon in polygons:
                drawer.polygon(polygon, outline=255, fill=255)
            # Show the mask for extra debugging
            # mask.show()
            
            # Rotate the mask
            # mask = transforms.Compose([
            #     transforms.Resize((480, 640))
            # ])(mask)
            mask = transforms.functional.rotate(mask, random_angle)

            mask = np.array(mask) == 255
            if DEBUG:
                print(f"Label '{label}' has {np.sum(mask)} pixels. Assigning them a value {self.SEGM_LABELS[label]['id']}")
            
            # Merge three road classes into one to improve the performance of the model
            if label in ['Ego Lane', 'Opposite Lane', 'Intersection']:
                target[mask] = self.SEGM_LABELS['Ego Lane']['id']
            else:
                target[mask] = self.SEGM_LABELS[label]['id']
        
        img = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.Resize((480, 640)),
            transforms.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.2),
            # Normalize the image with the mean and standard deviation of the ImageNet dataset
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)
        img = transforms.functional.rotate(img, random_angle)
        
        target = torch.from_numpy(target)
        
        # Cut the top 30% of the image and the target to remove the far-away parts of the scene
        img = img[:, (64 * 3):, :]
        target = target[(64 * 3):, :]
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def label_img_to_rgb(label_img):
        """
        Converts a label image (with one channel per label) to an RGB image.
        """
        rgb_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
        for label, label_info in DTSegmentationDataset.SEGM_LABELS.items():
            mask = label_img == label_info['id']
            rgb_img[mask] = label_info['rgb_value']
        return rgb_img


# ---------------------
# Randomly select a batch of images and masks from the dataset 
# and visualize them to check if the dataloader works correctly

if __name__ == "__main__":
    if DEBUG:
        dataset = DTSegmentationDataset()
        image, target = dataset[0]
        print(image.shape)
        transforms.ToPILImage()(image).show()
        transforms.ToPILImage()(DTSegmentationDataset.label_img_to_rgb(target)).show()