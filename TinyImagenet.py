import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        if split == "train":
            self.data_dir = os.path.join(root, "train")
            self.image_paths, self.labels = self._load_train_data()
        elif split == "val":
            self.data_dir = os.path.join(root, "val/images")
            self.image_paths, self.labels = self._load_val_data()
        elif split == "test":
            self.data_dir = os.path.join(root, "test/images")
            self.image_paths = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir)]
            self.labels = None  # No labels in test set

    def _load_train_data(self):
        image_paths = []
        labels = []
        wnid_to_label = {wnid: i for i, wnid in enumerate(os.listdir(self.data_dir))}

        for wnid in wnid_to_label.keys():
            class_dir = os.path.join(self.data_dir, wnid, "images")
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(wnid_to_label[wnid])

        return image_paths, labels

    def _load_val_data(self):
        image_paths = []
        labels = []
        val_annotations = os.path.join(self.root, "val/val_annotations.txt")
        wnid_to_label = {wnid: i for i, wnid in enumerate(open(os.path.join(self.root, "wnids.txt")).read().splitlines())}

        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name, wnid = parts[0], parts[1]
                image_paths.append(os.path.join(self.data_dir, image_name))
                labels.append(wnid_to_label[wnid])

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx] if self.labels else -1  # -1 for test set

        if self.transform:
            image = self.transform(image)

        return image, label
