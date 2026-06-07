"Extracted from https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm"

import os

import numpy as np
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.utils import save_image
import torchvision

import torch.utils.data as data


class ImageFolderDataset(data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")  # always convert to RGB
        if self.transform:
            image = self.transform(image)
        return image   # no labels returned


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data',env='train', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    if env in ['train', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, [target,color_red,color_green] = self.data_label_tuples[index]
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, [target,color_red,color_green]

  def __len__(self):
    return len(self.data_label_tuples)


class CelebAHQ_imgonly(data.Dataset):
  def __init__(self, img_root, anno_path, transform=None):
    self.img_root = img_root
    self.anno_path = anno_path

    # only need annotations file to get the list of train/test images
    # actual annotations won't be used
    self.label_dict, self.num_images = self.load_annotations(self.anno_path)
    self.transform = transform

  def __len__(self):
    return self.num_images

  def load_annotations(self, anno_path):
    with open(anno_path, 'r') as f:
      lines = f.read().splitlines()

    lines = [line.split(' ') for line in lines]

    # # there is one extra line, removing it (removed it manually)
    # lines = lines[1:]
    for idx in range(len(lines)):
      if '' in lines[idx]:
        lines[idx].remove('')

    # saving only the image filename
    label_dict = {}
    # one line is extra (has class names)
    for img_idx in range(len(lines) - 1):
      curr_line = lines[img_idx + 1]
      label_dict[img_idx] = [f'{curr_line[0]}']

    return label_dict, len(lines) - 1

  def __getitem__(self, index):
    curr_img_path = self.label_dict[index][0]
    image = Image.open(os.path.join(self.img_root, curr_img_path))

    if self.transform is not None:
      image = self.transform(image)

    return image


class CelebAHQ_dataset(data.Dataset):
  def __init__(self, img_root, anno_path, set_of_classes=None, transform=None):
    self.set_of_classes = set_of_classes
    self.img_root = img_root
    self.anno_path = anno_path

    self.label_dict, self.num_images = self.load_annotations(self.anno_path)
    self.transform = transform

  def __len__(self):
    return self.num_images

  def load_annotations(self, anno_path):
    with open(anno_path, 'r') as f:
      lines = f.read().splitlines()

    lines = [line.split(' ') for line in lines]

    # # there is one extra line, removing it (removed it manually)
    # lines = lines[1:]
    for idx in range(len(lines)):
      if '' in lines[idx]:
        lines[idx].remove('')

    # since first value is image name, so we take index+1 for the class labels
    cls_idx = [(lines[0].index(curr_cls) + 1) for curr_cls in self.set_of_classes]
    # print(cls_idx)

    label_dict = {}
    # one line is extra (has class names)
    for img_idx in range(len(lines) - 1):
      curr_line = lines[img_idx + 1]
      label_dict[img_idx] = [f'{curr_line[0]}']
      label_dict[img_idx] += [curr_line[lbl_idx] for lbl_idx in cls_idx]

    return label_dict, len(lines) - 1

  def __getitem__(self, index):
    curr_img_path = self.label_dict[index][0]
    image = Image.open(os.path.join(self.img_root, curr_img_path))
    label = self.label_dict[index][1:]
    label = np.asarray(label, dtype=np.int8)
    label[label == -1] = 0

    # only supports one set of classes at a time!!
    #if len(self.set_of_classes) > 1:
    #  label = np.argmax(label)
    #else:
      # print(label.shape) ### shape is (1,) so we just take the value instead of keeping as an array
    #  label = label[0]

    if self.transform is not None:
      image = self.transform(image)

    return image, label


class CelebAHQ_dataset_multiconc(data.Dataset):
  def __init__(self, img_root, anno_path, set_of_classes=None, transform=None):
    self.set_of_classes = set_of_classes
    self.img_root = img_root
    self.anno_path = anno_path

    self.label_dict, self.num_images = self.load_annotations(self.anno_path)
    self.transform = transform

  def __len__(self):
    return self.num_images

  def load_annotations(self, anno_path):
    with open(anno_path, 'r') as f:
      lines = f.read().splitlines()

    lines = [line.split(' ') for line in lines]

    # # there is one extra line, removing it (removed it manually)
    # lines = lines[1:]
    for idx in range(len(lines)):
      if '' in lines[idx]:
        lines[idx].remove('')

    # since first value is image name, so we take index+1 for the class labels
    cls_idx = [(lines[0].index(curr_cls) + 1) for curr_cls in self.set_of_classes]
    # print(cls_idx)

    label_dict = {}
    # one line is extra (has class names)
    for img_idx in range(len(lines) - 1):
      curr_line = lines[img_idx + 1]
      label_dict[img_idx] = [f'{curr_line[0]}']
      label_dict[img_idx] += [curr_line[lbl_idx] for lbl_idx in cls_idx]

    return label_dict, len(lines) - 1

  def __getitem__(self, index):
    curr_img_path = self.label_dict[index][0]
    image = Image.open(os.path.join(self.img_root, curr_img_path))
    label = self.label_dict[index][1:]
    label = np.asarray(label, dtype=np.int8)
    label[label == -1] = 0

    label = label.tolist()

    if self.transform is not None:
      image = self.transform(image)

    return image, label
  
class CUB_dataset_multiconc(data.Dataset):
  '''
  img_root: path to folder containing images (/expanse/lustre/projects/ddp390/akulkarni/datasets/CUB_200_2011/images)
  image_path: path to file containing image ID to image path list (/expanse/lustre/projects/ddp390/akulkarni/datasets/CUB_200_2011/images.txt)
  anno_path: path to file containing image ID and attribute labels (/expanse/lustre/projects/ddp390/akulkarni/datasets/CUB_200_2011/attributes/image_attribute_labels.txt)
  set_of_classes has to be a list of attribute IDs from CUB
  '''
  def __init__(self, img_root, image_path, anno_path, split_path, set_of_classes=None, transform=None, label_format='list', split='train', tipzs=True):
    self.set_of_classes = set_of_classes
    self.img_root = img_root
    self.anno_path = anno_path
    self.image_path = image_path
    self.label_format = label_format
    self.split = split
    self.split_path = split_path
    assert self.split in ['train', 'test']
    self.tipzs = tipzs

    # this has the actual number of images in the train or test split (whichever is specified)
    self.split_img_ids = self.load_train_test_split(self.split_path)
    self.num_images = len(self.split_img_ids)

    # these will load all image paths and all labels (not just train/test separately), we will take the image id based on above
    self.label_dict, self.total_num_images, self.image_attributes = self.load_annotations(self.anno_path, imgs_split=self.split_img_ids)
    self.image_dict = self.load_image_paths(self.image_path)
    # self.image_ids = list(self.image_dict.keys())
    # print(self.num_images, self.label_dict.shape) # 11788
    self.transform = transform

  def __len__(self):
    return self.num_images

  # modified from GPT4 generated code
  def load_image_paths(self, file_path):
    # Initialize an empty dictionary to store image paths
    image_paths = {}
    # Open and read the file
    with open(file_path, 'r') as file:
      for line in file:
        # Split each line into image_id and path
        parts = line.strip().split(maxsplit=1)
        image_id = int(parts[0])
        image_path = parts[1]
        # Store the path in the dictionary using image_id as the key
        image_paths[image_id] = image_path
    return image_paths

  def load_train_test_split(self, split_path):
    img_ids = []
    with open(split_path, 'r') as f:
      for line in f:
        img_id, split_indicator = line.strip().split()
        img_id = int(img_id)
        split_indicator = int(split_indicator)

        if (split_indicator == 1 and self.split == 'train') or (split_indicator == 0 and self.split == 'test'):
          img_ids.append(img_id)
    return img_ids

  # modified from GPT4 generated code
  def load_annotations(self, anno_path, imgs_split=None):
    # Create a dictionary to store the attributes for each image
    image_attributes = defaultdict(dict)
    
    # Read the image attribute labels file
    with open(anno_path, 'r') as file:
      for line in file:
        parts = line.strip().split()
        image_id = int(parts[0])
        attribute_id = int(parts[1])
        is_present = int(parts[2])
        
        # Store the presence label only for the specified attribute IDs
        if (imgs_split is None) and  (attribute_id in self.set_of_classes):
          image_attributes[image_id][attribute_id] = is_present
        if (imgs_split is not None):
          if (attribute_id in self.set_of_classes) and (image_id in imgs_split):
            image_attributes[image_id][attribute_id] = is_present
    
    # Determine the total number of images
    #total_images = max(image_attributes.keys())# + 1
    total_images = len(image_attributes)
    # Create the binary presence matrix
    presence_matrix = np.zeros((total_images, len(self.set_of_classes)), dtype=int)

    # image IDs are 1-indexed, so we subtract 1 while saving in the 0-indexed matrix
    # Fill the matrix with presence labels
    #for image_id, attributes in image_attributes.items():
    for i in range(len(self.split_img_ids)):
      attributes = image_attributes[self.split_img_ids[i]]
      for idx, attribute_id in enumerate(self.set_of_classes):
        presence_matrix[i, idx] = attributes.get(attribute_id, 0)
    
    return presence_matrix, total_images, image_attributes

  def __getitem__(self, index):
    image_id = self.split_img_ids[index]
    # image_id = self.image_ids[index]
    curr_img_path = os.path.join(self.img_root, self.image_dict[image_id])
    image = Image.open(curr_img_path).convert('RGB')
    # using index here since label dict is 0-indexed but image_dict was 1-indexed
    label = self.label_dict[index]
    #label = self.label_dict[image_id - 1]
    label = np.asarray(label, dtype=np.uint8)
    if self.label_format == 'list':
      label = label.tolist()

    ### just take the value if only one class instead of having (1,) tensor
    #### may have to recheck if this change works for training supervised evaluation classifiers, did this for TIP pseudo-labeling
    if len(self.set_of_classes) == 1 and self.tipzs:
      #label = label[0]
      label = self.image_attributes[image_id][self.set_of_classes[0]]
    if self.transform is not None:
      image = self.transform(image)

    if self.label_format == 'none':
      return image
    else:
      return image, label
