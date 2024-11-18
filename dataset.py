import os
import torch
from typing import Any
from torchvision import transforms, datasets
from datasets import load_dataset


class DualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform1=None, transform2=None):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        # label = self.dataset[index]['label']
        label = torch.tensor(self.dataset[index]['label'], dtype=torch.long)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform1 is not None:
            image1 = self.transform1(image)
        if self.transform2 is not None:
            image2 = self.transform2(image)
        return image1, image2, label
    
    def __len__(self):
        return len(self.dataset) 


def custom_collate(batch):
    images1 = torch.stack([item[0] for item in batch])
    images2 = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return images1, images2, labels


def get_simple_transformation():
    return transform1_simple, transform2_simple

def get_hard_transformation():
    return transform1_hard, transform2_hard


transform1_simple= transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

transform2_simple = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

transform1_hard = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(size=224, padding=4, pad_if_needed=True),
    transforms.ToTensor(),
])

transform2_hard = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.Resize(224),
    transforms.RandomCrop(size=224, padding=4, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])


if __name__ == "__main__":
    data_path = 'F:/ImageClassifier/PyRamidFER/data/RAFDB'
    dataset = load_dataset(path=data_path)
    print(dataset)
    train_data = DualDataset(dataset=dataset['validation'], transform1=transform1_simple, transform2=transform2_simple)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, collate_fn=custom_collate)
    for i, (images1, images2, labels) in enumerate(train_loader):
        print(images1.shape), print(images2.shape), print(labels)
        break