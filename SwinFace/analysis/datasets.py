import json
import torch
import numpy as np
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import scipy.io
import csv


def _rescale_landmark_coords(label, image_size, target_size):
    """Normalize landmark coordinates using the actual resize factor.

    If the original image is resized from (W, H) to target_size, then the
    normalized landmark coordinates in the resized image space are simply
    x / W and y / H.  This keeps the labels aligned with the image resize
    geometry without assuming the raw labels were already in the target space.
    """
    coords = np.asarray(label, dtype=np.float32).reshape(-1, 2)
    width, height = image_size
    scale_x = target_size / width
    scale_y = target_size / height

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    coords[:, 0] *= scale_x / target_size
    coords[:, 1] *= scale_y / target_size

    return coords.reshape(-1)


class _CelebADataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):

        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.target_size = int(config.img_size)

        self.CelebA_data = config.CelebA_data
        if choose == "train":
            self.CelebA_label = config.CelebA_train_label
        elif choose == "val":
            self.CelebA_label = config.CelebA_val_label
        elif choose == "test":
            self.CelebA_label = config.CelebA_test_label

        with open(self.CelebA_label, "r") as f:
            data = f.readlines()  

            for i, raw_line in enumerate(data):
                line = raw_line.strip().split()
                if len(line) < 11:  # image_name + 10 landmark points
                    continue
                
                image_path = os.path.join(self.CelebA_data, line[0])
                self.image_paths.append(image_path)
                raw_label = np.asarray([float(x) for x in line[1:11]], dtype=np.float32)
                self.labels.append(raw_label)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = _rescale_landmark_coords(self.labels[idx], original_size, self.target_size)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.image_paths)
    
class _MTFLDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):

        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.target_size = int(config.img_size)

        if choose == "train":
            self.MTFL_data = config.MTFL_train_data
            self.MTFL_label = config.MTFL_train_label
        elif choose == "test":
            self.MTFL_data = config.MTFL_test_data
            self.MTFL_label = config.MTFL_test_label

        with open(self.MTFL_label, "r") as f:
            data = f.readlines()  

            for i, raw_line in enumerate(data):
                line = raw_line.strip().split()
                if len(line) < 11:  # image_name + 10 landmark points
                    continue

                image_path = os.path.join(self.MTFL_data, line[0])
                self.image_paths.append(image_path)

                raw_label = np.asarray([float(x) for x in line[1:11]], dtype=np.float32)
                self.labels.append(raw_label)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = _rescale_landmark_coords(self.labels[idx], original_size, self.target_size)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.image_paths)
    
class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.target_size = int(config.img_size)
        celeba_dataset = _CelebADataset(config=config, choose=choose, transform=transform)
        mtfl_dataset = _MTFLDataset(config=config, choose=choose, transform=transform)

        self.image_paths.extend(celeba_dataset.image_paths)
        self.labels.extend(celeba_dataset.labels)
        self.image_paths.extend(mtfl_dataset.image_paths)
        self.labels.extend(mtfl_dataset.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = _rescale_landmark_coords(self.labels[idx], original_size, self.target_size)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.labels)

    

class _RAFDBDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if choose == 'train':
            self.RAF_data = config.RAF_data_train
            self.RAF_label = config.RAF_label_train
        elif choose == 'test':
            self.RAF_data = config.RAF_data_test
            self.RAF_label = config.RAF_label_test

        with open(self.RAF_label, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                image_name = row[0]
                label = int(row[1]) - 1
                image_path = os.path.join(self.RAF_data, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)


class _AffectNetDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if choose == 'train':
            self.AffectNet_data = config.AffectNet_train_data
            self.AffectNet_label = config.AffectNet_train_label
        elif choose == 'test':
            self.AffectNet_data = config.AffectNet_val_data
            self.AffectNet_label = config.AffectNet_val_label

        ann_files = sorted(os.listdir(self.AffectNet_label))
        for ann in ann_files:
            ann_path = os.path.join(self.AffectNet_label, ann)
            try:
                with open(ann_path, 'r') as af:
                    info = json.load(af)
            except Exception:
                continue

            # human-label is the expression label
            lbl = int(info['human-label'])
            if lbl > 6:  # AffectNet has 8 classes, but we only use 7 (ignore 'Contempt')
                continue

            img_name = ann[:-5] + '.jpg'
            image_path = os.path.join(self.AffectNet_data, img_name)
            if not os.path.exists(image_path):
                print(f"Error: Image not found for AffectNet annotation {ann}")
                continue
            self.image_paths.append(image_path)
            self.labels.append(lbl)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)


class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        raf_dataset = _RAFDBDataset(config=config, choose=choose, transform=transform)
        affectnet_dataset = _AffectNetDataset(config=config, choose=choose, transform=transform)

        self.image_paths.extend(raf_dataset.image_paths)
        self.labels.extend(raf_dataset.labels)
        self.image_paths.extend(affectnet_dataset.image_paths)
        self.labels.extend(affectnet_dataset.labels)

        self.weight = dict()
        self.weight["RAF"] = 0.5
        self.weight["AffectNet"] = 0.5

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)


class _AFLWDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        """
        Reads from a single annotation file with format per line:
        image_name landmark_x1 landmark_y1 ... landmark_x21 landmark_y21 bbox_x bbox_y bbox_w bbox_h roll pitch yaw
        """
        self.image_paths = []
        self.labels = []  # [pose (3,)]
        self.transform = transform

        if choose == "train":
            self.AFLW_data = config.AFLW_train_data
            self.AFLW_label = config.AFLW_train_label
        elif choose == "val":
            self.AFLW_data = config.AFLW_val_data
            self.AFLW_label = config.AFLW_val_label
        elif choose == "test":
            self.AFLW_data = config.AFLW_test_data
            self.AFLW_label = config.AFLW_test_label

        # Read annotation file
        with open(self.AFLW_label, "r") as f:
            data = f.readlines()

        for i, raw_line in enumerate(data):
            line = raw_line.strip().split()
            if len(line) < 46:  # image_name + 45 values minimum
                continue

            image_name = line[0]
            try:
                pose = np.array([float(line[j]) for j in range(43, 46)], dtype=np.float32)
                self.image_paths.append(os.path.join(self.AFLW_data, image_name))
                self.labels.append(pose)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {i}: {line[:6]}... Error: {e}")
                continue

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.image_paths)

class _LP300WDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        """Load 300W-LP images and annotations for head pose and landmark regression."""
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if choose == "train":
            self.LP300W_root = config.WLP_data_train
        elif choose == "test":
            self.LP300W_root = config.WLP_data_test

        if not os.path.isdir(self.LP300W_root):
            raise ValueError(f"Invalid 300W_LP root path: {self.LP300W_root}")

        valid_dirs = {
            "AFW",
            "HELEN",
            "IBUG",
            "LFPW",
        }

        for subset in sorted(os.listdir(self.LP300W_root)):
            subset_dir = os.path.join(self.LP300W_root, subset)
            if subset not in valid_dirs or not os.path.isdir(subset_dir):
                continue

            for file_name in sorted(os.listdir(subset_dir)):
                if not file_name.lower().endswith('.mat'):
                    continue

                mat_path = os.path.join(subset_dir, file_name)
                img_name = os.path.splitext(file_name)[0] + '.jpg'
                image_name = os.path.join(subset, img_name)
                image_path = os.path.join(self.LP300W_root, image_name)
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found for 300W_LP annotation {mat_path}")
                    continue

                try:
                    mat = scipy.io.loadmat(mat_path)
                except Exception as e:
                    print(f"Warning: Failed to load mat file {mat_path}: {e}")
                    continue

                pose_para = mat.get('Pose_Para')
                if pose_para is None:
                    print(f"Warning: Missing pt2d or Pose_Para in {mat_path}")
                    continue

                pose = np.asarray(pose_para, dtype=np.float32).squeeze()
                pose = pose.reshape(-1)
                if pose.size < 3:
                    print(f"Warning: Unexpected Pose_Para shape {pose_para.shape} in {mat_path}")
                    continue
                pose = pose[:3]

                self.image_paths.append(image_path)
                self.labels.append(pose)

        if len(self.image_paths) == 0:
            raise ValueError(f"No 300W_LP samples found in {self.LP300W_root}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.image_paths)


class HeadPoseDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        lp300w_dataset = _LP300WDataset(config=config, choose=choose, transform=transform)
        aflw_dataset = _AFLWDataset(config=config, choose=choose, transform=transform)

        self.image_paths.extend(lp300w_dataset.image_paths)
        self.labels.extend(lp300w_dataset.labels)
        self.image_paths.extend(aflw_dataset.image_paths)
        self.labels.extend(aflw_dataset.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)
