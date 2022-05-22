# Anno contains identity_CelebA.txt which contains the file name and label for the identity of the person
# Ill need to go through these and find "all the matching people" i guess
# N tasks, each containing n "families", each family has "k" copies of themselves

# from sklearn.datasets import fetch_lfw_people
# from sklearn import datasets
# import torchvision.datasets as datasets

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from textwrap import wrap

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from PIL import Image

# lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)
# lfw_people2 = fetch_lfw_people(min_faces_per_person=1, resize=0.4)

workers = 4
ngpu = 1

# User must manually download the celebA folder from https://drive.google.com/drive/folders/1ngzl-cI1s-Ib8k_e9LDVZ_eQJRHvXq2V (correct as of 17/05/2022)

# this must be the folder containing a subfolder of the images
dataroot = r"C:\Users\Rajesh\Documents\Pycharm Projects\Y4_COMP0138\Datasets\celebA\CelebA-20220516T115258Z-001\CelebA\Img\img_align_celeba"
# the folder containing the images itself
image_folder_name = r"\img_align_celeba"
# text file with the identity labels
labels_path = r"C:\Users\Rajesh\Documents\Pycharm Projects\Y4_COMP0138\Datasets\celebA\CelebA-20220516T115258Z-001\CelebA\Anno\identity_CelebA.txt"
# device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# imagenet_data = datasets.ImageFolder(
#     root=r"C:\Users\Rajesh\Documents\Pycharm Projects\Y4_COMP0138\Datasets\celebA\CelebA-20220516T115258Z-001\CelebA\Img\img_align_celeba")

# dataset = datasets.ImageFolder(root=dataroot,
#                                transform=transforms.Compose([
#                                    transforms.Resize(image_size),
#                                    transforms.CenterCrop(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))
#
# # Create the dataloader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=" ", names=['x', 'y'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = image.float()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MyDataset(Dataset):
    def __init__(self, dataset, class_idx):
        self.dataset = dataset
        self.mapping = torch.arange(len(dataset))[dataset.img_labels == class_idx]

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        return self.dataset[self.mapping[idx]]


def dataset_creation(N, n, k, dataroot, labelspath, image_size=64):
    if not os.path.exists('familydata'):
        os.makedirs('familydata')

    transformation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    new_dataset = CustomImageDataset(labelspath, dataroot + image_folder_name, transform=transformation)
    df = new_dataset.img_labels

    min_df = df.groupby('y').filter(lambda x: len(x) >= k)  # only consider people with >= k samples
    min_classes = min_df["y"].nunique()  # only consider when we have at least n*k classes
    min_data = len(min_df)

    if min_data < N * n * k:
        print("Not enough samples available")
        return
    if min_classes < N * n:
        print("Not enough classes possible")
        return

    unique, counts = np.unique(min_df.y.ravel(), return_counts=True)

    for i in range(N):
        print(f"beginning sampling for family #{i + 1}")
        im_array = torch.empty((image_size, k * image_size, 3))
        family_array = np.zeros(n)
        for x in range(n):
            label = np.random.choice(unique)
            unique = np.delete(unique, np.where(unique == label))
            train_idx = np.where((new_dataset.img_labels == label))[0]
            train_subset = Subset(new_dataset, train_idx)
            if len(np.unique(train_idx)) != len(train_idx):  # it should never hit this, duplicate image check
                print(f"why was there a duplicate for {train_idx}?")
                print(np.unique(train_idx))
            train_loader_subset = DataLoader(train_subset, shuffle=True, batch_size=k)  # get batches of the same person
            real_batch, labels = next(iter(train_loader_subset))
            im_array = torch.cat([im_array, torch.cat(real_batch.split(1, 0), 3).squeeze().permute(1, 2, 0)])
            family_array[x] = label
            # im = Image.fromarray(
            #     (torch.cat(real_batch.split(1, 0), 3).squeeze()).permute(1, 2, 0).numpy().astype('uint8'))
            # plt.imshow(im)
            # plt.title(f"family #{i + 1} member #{label}")
            # plt.show()
            # plt.clf()
            # print(f'Family member #{label} added')

        im_array = im_array[image_size:, :, :]
        image = Image.fromarray(im_array.numpy().astype('uint8'))

        string = ""
        for x in family_array:
            string += f"{int(x)}"
            if x != family_array[-1]:
                string += f", "

        plt.imshow(image)
        plt.title("\n".join(wrap(f"family #{i + 1} members {string}", 60)))
        plt.savefig(f"./familydata/family_{i + 1}.png")
        plt.show()
        plt.clf()


if __name__ == '__main__':
    # new_dataset = CustomImageDataset(labels, dataroot + "\img_align_celeba")
    # dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # real_batch, labels = next(iter(dataloader))
    # im = Image.fromarray(
    #     (torch.cat(real_batch.split(1, 0), 3).squeeze()).permute(1, 2, 0).numpy().astype('uint8'))
    # im.save("ground_truth_images2.jpg")
    # plt.imshow(im)
    # plt.show()
    # print('Example montage of 16 ground truth images have been saved to ground_truth_images.jpg saved.\n')

    # real_batch, labels = next(iter(dataloader))
    # img = Image.fromarray((torch.cat(real_batch.split(1, 0), 3).squeeze()).permute(1, 2, 0).numpy().astype('uint8'))
    # label = labels[0]
    # # plt.imshow(real_batch.squeeze().permute(1, 2, 0))
    # plt.imshow(img)
    # plt.show()
    # print(f'Example of {batch_size} images of random people.\n')

    # new_dataset = CustomImageDataset(labels, dataroot + "\img_align_celeba")
    # dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # column = new_dataset.img_labels["y"]
    # max_value = column.max()
    # # print(new_dataset.img_labels['y'].value_counts())
    # for i in range(1):
    #     train_idx = []
    #     while len(train_idx) < batch_size:
    #         label = np.random.randint(0, max_value)
    #         train_idx = np.where((new_dataset.img_labels == label))[0]
    #     train_subset = Subset(new_dataset, train_idx)
    #     train_loader_subset = DataLoader(train_subset, shuffle=True,
    #                                      batch_size=batch_size)  # this dataloader will get batches of the same person
    #     real_batch, labels = next(iter(train_loader_subset))
    #     im = Image.fromarray((torch.cat(real_batch.split(1, 0), 3).squeeze()).permute(1, 2, 0).numpy().astype('uint8'))
    #     plt.imshow(im)
    #     plt.show()
    #     print(f'Example of {batch_size} images of person #{label}.\n')

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_subset), size=(1,)).item()
    #     img, label = train_subset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     # plt.title('{}:{}'.format(label, labels_map[label]))
    #     plt.axis("off")
    #     plt.imshow(img.squeeze().permute(1, 2, 0))

    # # Plot some training images
    # real_batch = next(iter(train_loader_subset))
    # plt.figure(figsize=(batch_size, batch_size))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(utils.make_grid(real_batch[0].to(device)[:batch_size], padding=2).cpu(), (1, 2, 0)))
    # plt.show()

    # you have N number of tasks
    # Each task has n classes
    # Each class has k images of each item

    # You need to check that theres enough data for N * n * k samples
    # You need to check that theres N*n classes which have atleast k samples

    # column = new_dataset.img_labels["y"]
    # max_value = column.max()
    # print(new_dataset.img_labels['y'].value_counts())
    # dataset = new_dataset[new_dataset.img_labels.groupby('y').x.transform(len) > 10]
    # print(dataset.img_labels['y'].value_counts())

    # counts = new_dataset.img_labels['y'].value_counts()
    #
    # to_remove = counts[counts > 3].index
    # print(to_remove)
    #
    # df = new_dataset.img_labels[~new_dataset.img_labels.x.isin(to_remove)]
    #
    # print(df['y'].value_counts())

    # df = new_dataset.img_labels
    #
    # sub_df2 = df.groupby('y').filter(lambda x: len(x) > 10)
    # # this is ensuring everyone has atleast k samples, replace 3 with k
    # # print(sub_df2['y'].value_counts())
    #
    # print(sub_df2["y"].nunique())
    # # can compare this against N * n to ensure it is possible to get n people with k samples
    #
    # print(
    #     len(sub_df2))  # can compare this against N * n * k, if there are less than N * n * k then we dont have enough samples

    # column = sub_df2["y"]
    # max_value = column.max()
    # # print(new_dataset.img_labels['y'].value_counts())
    # for i in range(1):
    #     train_idx = []
    #     while len(train_idx) < batch_size:
    #         label = np.random.randint(0, max_value)
    #         train_idx = np.where((new_dataset.img_labels == label))[0]
    #     train_subset = Subset(new_dataset, train_idx)
    #     train_loader_subset = DataLoader(train_subset, shuffle=True,
    #                                      batch_size=batch_size)  # this dataloader will get batches of the same person
    #     real_batch, labels = next(iter(train_loader_subset))
    #     im = Image.fromarray((torch.cat(real_batch.split(1, 0), 3).squeeze()).permute(1, 2, 0).numpy().astype('uint8'))
    #     plt.imshow(im)
    #     plt.show()
    #     print(f'Example of {batch_size} images of person #{label}.\n')
    N_tasks = 5
    n_classes = 5
    k_samples = 5
    imagesize = 64
    dataset_creation(N_tasks, n_classes, k_samples, dataroot, labels_path, imagesize)
