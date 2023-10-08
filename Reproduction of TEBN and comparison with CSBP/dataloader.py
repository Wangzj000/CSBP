import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from spikingjelly.datasets import cifar10_dvs


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Altered from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
        img (Tensor): Tensor image of size (C, H, W).
        Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


# Load Cifar-10,100

def Cifar10(download=False):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                   std=[n / 255. for n in [68.2, 65.4, 70.4]])
        ])
        # Cutout(n_holes=1, length=16)])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                 std=[n / 255. for n in [68.2, 65.4, 70.4]])])
    train_dataset = CIFAR10(root='./datasets/CIFAR10', train=True, download=download, transform=transform_train)
    val_dataset = CIFAR10(root='./datasets/CIFAR10', train=False, download=download, transform=transform_test)
    return train_dataset, val_dataset


def Cifar100(download=False):
    transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])
            ])
         # Cutout(n_holes=1, length=16)])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])])
    train_dataset = CIFAR100(root='./datasets/CIFAR100', train=True, download=download, transform=transform_train)
    val_dataset = CIFAR100(root='./datasets/CIFAR100', train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset


# Load DVS-CIFAR10

class ToTensor(object):
    def __call__(self, pic):
        return torch.from_numpy(pic).float()

def DVSCifar10():
    transform_train = transforms.Compose([
        ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip()
    ])

    transform_test = transforms.Compose([
        ToTensor(),
        transforms.Resize(size=(48, 48))
    ])

    # train_set = cifar10_dvs.CIFAR10DVS(root='./datasets/CIFAR10DVS', train=True, data_type='frame', frames_number=10, split_by='number', transform=transform_train)
    # test_set = cifar10_dvs.CIFAR10DVS(root='./datasets/CIFAR10DVS', train=False, frames_number=10, split_by='number',  transform=transform_test)
    data = cifar10_dvs.CIFAR10DVS(root='./datasets/CIFAR10DVS', data_type='frame', frames_number=10, split_by='number',  transform=transform_test)

    return data

if __name__ == '__main__':
    sets = DVSCifar10()
