import os
import torch
import torch.utils.data
import torchvision
import random
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image, ExifTags, ImageOps, ImageEnhance

# initialisation
INPUT_SIZE = (572, 572)
OUTPUT_SIZE = (388, 388)
supercategories = np.array([[0], [1], [2, 3], [4, 5, 6], [7, 8], [9], [10, 11, 12], [13, 14, 15, 16, 17, 18, 19],
                            [20, 21, 22, 23, 24], [25], [26], [27, 28], [29], [30, 31, 32, 33], [34, 35],
                            [36, 37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47], [48], [49], [50], [51], [52],
                            [53], [54], [55, 56], [57], [58], [59]])

# two helper methods from the dataset
def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    return rle

# maps categories onto their super categories
def get_supercategory_id(annotation_id):
    i = 0
    for item in supercategories:
        if annotation_id in item:
            return i
        i += 1
    print("invalid id")


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


class Taco(torch.utils.data.Dataset):
    dataset_dir = "./data"
    ann_file_path = os.path.join(dataset_dir, 'annotations') + ".json"

    def __init__(self, root=dataset_dir, annotation=ann_file_path, transforms=get_transform(),
                 data_augmentation=False, super_categories=False):
        super(Taco, self).__init__()
        self.root = root
        self.transforms = transforms
        self.data_augmentation = data_augmentation
        self.data_augmentation_flag = 0
        self.super_categories = super_categories

        self.coco = COCO(annotation)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.broken_image_indices = self.get_broken_image_indices()

    def __getitem__(self, index):
        if self.data_augmentation:
            self.data_augmentation_flag = random.randint(0, 3)

        index = self.get_unbroken_image_index(index)
        image_id = self.image_ids[index]

        image = self.get_image(image_id)
        mask = self.get_mask(image_id)

        return image, mask

    def __len__(self):
        return len(self.image_ids)

    # runs through the annotations and finds broken pointers
    def get_broken_image_indices(self):
        broken_image_indices = []
        for image_id in self.image_ids:
            image_path = os.path.join(self.root, self.coco.loadImgs(image_id)[0]['file_name'])
            if not Path(image_path).is_file():
                broken_image_indices.append(image_id)
        return broken_image_indices

    # if a broken annotation is accessed a intact one is returned
    def get_unbroken_image_index(self, index):
        while index in self.broken_image_indices:
            index = random.randint(1, len(self.image_ids))
        return index

    def get_image(self, image_id):
        image_path = self.coco.loadImgs(image_id)[0]['file_name']
        with Image.open(os.path.join(self.root, image_path)) as image:

            # rotates images if needed
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            if image._getexif():
                exif = dict(image._getexif().items())
                # Rotate portrait and upside down images if necessary
                if orientation in exif:
                    if exif[orientation] == 3:
                        image = image.rotate(180, expand=True)
                    if exif[orientation] == 6:
                        image = image.rotate(270, expand=True)
                    if exif[orientation] == 8:
                        image = image.rotate(90, expand=True)

            # applies dada augmentation
            if self.data_augmentation_flag == 1:
                image = ImageOps.mirror(image)
            elif self.data_augmentation_flag == 2:
                image = ImageOps.flip(image)
            elif self.data_augmentation_flag == 3:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(np.random.uniform(0.35, 1.75))

            return self.transforms(image.resize(INPUT_SIZE))

    def get_mask(self, image_id):
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        h, w = self.coco.annToMask(annotations[0]).shape
        mask = np.zeros(self.coco.annToMask(annotations[0]).shape)

        # turns the annotations into masks and stacks them
        for annotation in annotations:
            if self.super_categories:
                mask += (get_supercategory_id(annotation["category_id"]) + 1) * annToMask(annotation, h, w)
            else:
                mask += (annotation["category_id"] + 1) * annToMask(annotation, h, w)
        mask_image = Image.fromarray(mask).resize(OUTPUT_SIZE, resample=0)

        # applies data augmentation
        if self.data_augmentation_flag == 1:
            mask_image = ImageOps.mirror(mask_image)
        if self.data_augmentation_flag == 2:
            mask_image = ImageOps.flip(mask_image)

        return torch.tensor(np.array(mask_image))

    # extra method for visualisation
    def get_example(self, index):
        resized_image, mask = self.__getitem__(index)
        image_id = self.image_ids[index]
        image_path = self.coco.loadImgs(image_id)[0]['file_name']
        with Image.open(os.path.join(self.root, image_path)) as full_image:
            original_image = self.transforms(full_image)

            return resized_image, mask, original_image
