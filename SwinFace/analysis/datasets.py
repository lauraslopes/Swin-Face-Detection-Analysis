import json
import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
import sqlite3
import scipy.io
import csv

class AgeGenderDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset=["IMDB", "WIKI", "Adience", "MORPH"], transform=None):

        self.root_path = config.age_gender_data_path

        self.image_paths = []
        self.labels = []
        
        self.suffix = ""

        self.sample_num = config.num_image // config.recognition_bz * config.age_gender_bz

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.weight = dict()
        self.weight["Adience"] = 0.25
        self.weight["MORPH"] = 0.25
        self.weight["IMDB+WIKI"] = 0.5

        flag = True
        self.labels_1 = []
        while flag:

            if "Adience" in dataset and flag:
                # Adience
                self.Adience_path = os.path.join(self.root_path, "Adience")
                self.Adience_data = os.path.join(self.Adience_path, "data")
                self.Adience_label = os.path.join(self.Adience_path, ("label" + self.suffix + ".txt"))

                with open(self.Adience_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.Adience_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_1.append(label)

                    if len(self.labels_1) >= self.sample_num * self.weight["Adience"]:
                        flag = False
                        break
        
        flag = True
        self.labels_2 = []
        while flag:

            if "MORPH" in dataset and flag:
                # MORPH
                self.MORPH_path = os.path.join(self.root_path, "MORPH")
                self.MORPH_data = os.path.join(self.MORPH_path, "data")
                self.MORPH_label = os.path.join(self.MORPH_path, ("label" + self.suffix + ".txt"))

                with open(self.MORPH_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.MORPH_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_2.append(label)

                    if len(self.labels_2) >= self.sample_num * self.weight["MORPH"]:
                        flag = False
                        break

        self.labels.extend(self.labels_1)
        self.labels.extend(self.labels_2)

        flag = True
        while flag:

            if "WIKI" in dataset and flag:
                # WIKI
                self.WIKI_path = os.path.join(self.root_path, "WIKI")
                self.WIKI_data = os.path.join(self.WIKI_path, "data")
                self.WIKI_label = os.path.join(self.WIKI_path, ("label" + self.suffix + ".txt"))

                with open(self.WIKI_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.WIKI_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

            if "IMDB" in dataset and flag:
                # IMDB
                self.IMDB_path = os.path.join(self.root_path, "IMDB")
                self.IMDB_data = os.path.join(self.IMDB_path, "data")
                self.IMDB_label = os.path.join(self.IMDB_path, ("label" + self.suffix + ".txt"))
                with open(self.IMDB_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):

                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.IMDB_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels.append(label)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_names = []
        self.labels = []
        random.seed(config.seed)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if choose == "train":
            self.CelebA_data = config.CelebA_train_data
            self.CelebA_label = config.CelebA_train_label

            self.sample_num = config.num_image // config.recognition_bz * config.CelebA_bz

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(1, len(data)):
                    r = random.random()
                    if r < 0.5:
                        line = data[i].split()

                        self.image_names.append(line[0])

                        label = [0 for j in range(40)]
                        for j in range(1, 41):
                            if line[j] == "1":
                                label[j-1] = 1

                        self.labels.append(label)

                        if len(self.labels) == self.sample_num:
                            flag = False
                            break
        elif choose == "val":
            self.CelebA_data = config.CelebA_val_data
            self.CelebA_label = config.CelebA_val_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()

                self.image_names.append(line[0])

                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1

                self.labels.append(label)

        elif choose == "test":
            self.CelebA_data = config.CelebA_test_data
            self.CelebA_label = config.CelebA_test_label

            with open(self.CelebA_label, "r") as f:
                data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split()

                self.image_names.append(line[0])

                label = [0 for j in range(40)]
                for j in range(1, 41):
                    if line[j] == "1":
                        label[j - 1] = 1

                self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.CelebA_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)

class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_paths = []
        self.labels = []

        if choose == 'train':
            self.RAF_data = config.RAF_data_train
            self.RAF_label = config.RAF_label_train
            self.AffectNet_data = config.AffectNet_train_data
            self.AffectNet_label = config.AffectNet_train_label
        elif choose == 'test':
            self.RAF_data = config.RAF_data_test
            self.RAF_label = config.RAF_label_test
            self.AffectNet_data = config.AffectNet_val_data
            self.AffectNet_label = config.AffectNet_val_label

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.weight = dict()
        self.weight["RAF"] = 0.5
        self.weight["AffectNet"] = 0.5
        
        # AffectNet+
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

            # determine image filename: try fields or derive from annotation filename
            img_name = ann[:-5] + '.jpg'
            image_path = os.path.join(self.AffectNet_data, img_name)
            if not os.path.exists(image_path):
                print(f"Error: Image not found for AffectNet annotation {ann}")
                continue
            self.image_paths.append(image_path)
            self.labels.append(lbl)
         
        # RAF
        with open(self.RAF_label, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                image_name = row[0]
                label = int(row[1]) - 1
                # images are in train/ or test/ subfolders inside RAF_DB
                
                image_path = os.path.join(self.RAF_data, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)
        
        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class RAFDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_paths = []
        self.labels = []

        # pick RAF label file depending on requested split
        if choose == 'train':
            self.RAF_data = config.RAF_data_train
            self.RAF_label = config.RAF_label_train
        elif choose == 'test':
            self.RAF_data = config.RAF_data_test
            self.RAF_label = config.RAF_label_test

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

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

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])

        return img, label

    def __len__(self):
        return len(self.image_paths)


class FGnetDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="all", id=0, transform=None):

        self.FGnet_data = config.FGnet_data
        self.FGnet_label = config.FGnet_label
        self.leave_out_file_name = ""

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        with open(self.FGnet_label, "r") as f:
            data = f.readlines()

        cut = len(data) // 10 * 9
        if choose == "remove_one":
            for i in range(1, len(data)):
                line = data[i].split()
                if i == id:
                    self.leave_out_file = os.path.join(self.FGnet_data, line[0])
                else:
                    self.image_names.append(line[0])
                    label = np.asarray([float(line[1])])
                    self.labels.append(label)
        elif choose == "all":
            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "9_fold":
            for i in range(1, cut):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "1_fold":
            for i in range(cut, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.FGnet_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])

        return img, label

    def __len__(self):
        return len(self.image_names)

    def get_leave_out_file_name(self):
        return self.leave_out_file_name
        
class LAPDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="", transform=None):

        if choose == "train":
            self.LAP_data = config.LAP_train_data
            self.LAP_label = config.LAP_train_label
        elif choose == "test":
            self.LAP_data = config.LAP_test_data
            self.LAP_label = config.LAP_test_label

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


        with open(self.LAP_label, "r") as f:
            data = f.readlines()
        

        for i in range(0, len(data)):
            line = data[i].split(";")
    
            self.image_names.append(line[0])
    
            label = [np.asarray([float(line[1])]), np.asarray([float(line[2])])]
    
            self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.LAP_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)

class AFLWDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="train", transform=None):
        """
        AFLW Dataset for head pose estimation and landmark detection.
        Reads from a single annotation file with format per line:
        image_name landmark_x1 landmark_y1 ... landmark_x21 landmark_y21 bbox_x bbox_y bbox_w bbox_h roll pitch yaw
        
        Total per line: image_name + 42 landmark coords + 3 pose angles = 46 numeric values
        """
        self.image_paths = []
        self.labels = []  # [landmarks (42,), pose (3,)]

        if choose == "train":
            self.AFLW_data = config.head_pose_train_data
            self.AFLW_label = config.head_pose_train_label
        elif choose == "val":
            self.AFLW_data = config.head_pose_val_data
            self.AFLW_label = config.head_pose_val_label
        elif choose == "test":
            self.AFLW_data = config.head_pose_test_data
            self.AFLW_label = config.head_pose_test_label
        else:
            raise ValueError(f"choose must be 'train', 'val', or 'test', got {choose}")

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Read annotation file
        with open(self.AFLW_label, "r") as f:
            data = f.readlines()

        for i, raw_line in enumerate(data):
            line = raw_line.strip().split()
            if len(line) < 46:  # image_name + 45 values minimum
                continue

            image_name = line[0]
            try:
                landmarks = np.array([float(line[j]) for j in range(1, 43)], dtype=np.float32)
                pose = np.array([float(line[j]) for j in range(43, 46)], dtype=np.float32)
                label = np.concatenate([landmarks, pose])
                self.image_paths.append(os.path.join(self.AFLW_data, image_name))
                self.labels.append(label)
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


class LP300WDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):
        """Load 300W-LP images and annotations for head pose and landmark regression."""
        self.image_paths = []
        self.labels = []

        self.LP300W_root = config.W_LP_data
        if not os.path.isdir(self.LP300W_root):
            raise ValueError(f"Invalid 300W_LP root path: {self.LP300W_root}")

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

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
                if not file_name.lower().endswith('0.mat'):
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

                pt2d = mat.get('pt2d')
                pose_para = mat.get('Pose_Para')
                if pt2d is None or pose_para is None:
                    print(f"Warning: Missing pt2d or Pose_Para in {mat_path}")
                    continue

                pt2d = np.asarray(pt2d, dtype=np.float32)
                if pt2d.shape == (2, 21):
                    landmarks = pt2d.T.reshape(-1)
                else:
                    landmarks = pt2d.reshape(-1)
                if landmarks.size != 42:
                    print(f"Warning: Unexpected pt2d shape {pt2d.shape} in {mat_path}")
                    continue

                pose = np.asarray(pose_para, dtype=np.float32).squeeze()
                pose = pose.reshape(-1)
                if pose.size < 3:
                    print(f"Warning: Unexpected Pose_Para shape {pose_para.shape} in {mat_path}")
                    continue
                pose = pose[:3]

                label = np.concatenate([landmarks, pose], axis=0)
                self.image_paths.append(image_path)
                self.labels.append(label)

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
    def __init__(self, config, choose, transform=None):

        self.image_paths = []
        self.labels = []

        self.image_paths.extend(LP300WDataset(config=config, transform=transform).image_paths)
        self.labels.extend(LP300WDataset(config=config, transform=transform).labels)
        self.image_paths.extend(AFLWDataset(config=config, choose=choose, transform=transform).image_paths)
        self.labels.extend(AFLWDataset(config=config, choose=choose, transform=transform).labels)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)
