import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import csv
import cv2
from line_profiler_pycharm import profile

from textwrap import wrap

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from PIL import Image

# dataroot = r"./CelebA/Img/img_align_celeba/img_align_celeba"
# labels_path = r"./CelebA/Anno/identity_CelebA.txt"

dataroot = r"./CelebA-20220516T115258Z-001/CelebA/Img/img_align_celeba/img_align_celeba"
labels_path = r"./CelebA-20220516T115258Z-001/CelebA/Anno/identity_CelebA.txt"
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class ListDict(dict):
    """ Dictionary whose values are lists. """

    def __missing__(self, key):
        value = self[key] = []
        return value


class CustomDataset(Dataset):

    def create_class_map(self, size):
        lstdct = ListDict()
        with open(self.label_path, 'r') as csvfile:
            for row in csv.reader(csvfile, delimiter=' '):
                value, key = row[:2]
                if not lstdct[key] or len(lstdct[key]) < size:
                    lstdct[key].append(value)
        return lstdct

    def get_key(self, class_name):
        for key, class_id in self.class_map.items():
            if class_name in class_id:
                return key

    def task_class_remap(self):
        # you can use this code to remap the classes and assign to tasks
        existing_mapping = self.class_map
        tasks = np.arange(0, self.tasks)
        task_map = {}
        # self.data = []
        for i, task in enumerate(tasks):  # for each task
            classes = np.arange(0, self.classes_per_task)
            class_map = {}
            keys_to_be_deleted = []
            for y, (class_remapped, (key, val)) in enumerate(zip(classes, existing_mapping.items())):
                class_map[y + i * self.classes_per_task] = val
                for each in val:
                    self.data.append([dataroot + os.path.sep + each, y + i * self.classes_per_task])
                keys_to_be_deleted.append(key)
            for key_ in keys_to_be_deleted:
                del (existing_mapping[key_])
            task_map[task] = class_map
        return task_map

    def __init__(self, tasks=3, classes=15, samples_per_class=10, img_path=dataroot, label_path=labels_path,
                 transform=None, image_size=32):
        self.img_dim = (image_size, image_size)
        self.transform = transform
        self.tasks = tasks  # 3
        self.classes = classes  # 15
        self.classes_per_task = int(classes / tasks)  # 5
        self.samples_per_class = samples_per_class  # 10
        self.samples_per_task = self.classes_per_task * self.samples_per_class  # 50

        self.img_path = img_path
        self.label_path = label_path
        # this creates a dict of lists, each key is the class, each value is list of files corresponding to that class
        self.class_map = self.create_class_map(self.samples_per_class)
        # trims the lists down
        self.class_map = {key: val for key, val in self.class_map.items() if len(val) >= self.samples_per_class}
        count_above_10 = len({key: val for key, val in self.class_map.items() if len(val) == self.samples_per_class})
        self.data = []
        self.task_map = self.task_class_remap()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img_tensor = cv2.resize(cv2.imread(img_path), self.img_dim)
        class_id = torch.tensor([int(class_id)])
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, class_id


# class CustomLoader:
#     def segment_dataset(self):
#         for task in range(self.dataset.tasks):
#             indexes = np.arange(task * self.number_of_samples, (task + 1) * self.number_of_samples)
#             self.train_indexes.append(np.random.choice(indexes, self.train_size, replace=False))
#             indexes = np.delete(indexes, np.where(np.isin(indexes, self.train_indexes)))
#             self.val_indexes.append(np.random.choice(indexes, self.val_size, replace=False))
#             indexes = np.delete(indexes, np.where(np.isin(indexes, self.val_indexes)))
#             self.test_indexes.append(np.random.choice(indexes, self.test_size, replace=False))
#             # indexes = np.delete(indexes, np.where(np.isin(indexes, self.test_indexes)))
#
#     def __init__(self, celeb_dataset):
#         self.dataset = celeb_dataset
#         self.number_of_samples = self.dataset.classes * self.dataset.samples_per_class
#         self.test_size = int(np.floor(self.number_of_samples / self.dataset.class_size))
#         self.train_size = int((self.number_of_samples - self.test_size) * .8)
#         self.val_size = self.number_of_samples - self.test_size - self.train_size
#         self.train_indexes = []
#         self.val_indexes = []
#         self.test_indexes = []
#         self.segment_dataset()
#         self.task_train = 0
#         self.task_val = 0
#         self.task_test = 0
#
#     def train_loader(self, current_task, batch_size=4):
#         train_subset = Subset(self.dataset, self.train_indexes[current_task])
#         train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
#         return train_loader
#
#     def val_loader(self, current_task, batch_size=4):
#         val_subset = Subset(self.dataset, self.val_indexes[current_task])
#         val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
#         return val_loader
#
#     def test_loader(self, current_task, batch_size=4):
#         test_subset = Subset(self.dataset, self.test_indexes[current_task])
#         test_loader = DataLoader(test_subset, shuffle=False, batch_size=batch_size)
#         return test_loader
#
#     def train_sampler(self, batch_size=4):
#         if self.dataset.tasks >= self.task_train:
#             train_subset = Subset(self.dataset, self.train_indexes[self.task_train])
#             train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
#             next_sample = next(iter(train_loader))
#             if train_loader.__len__() == 0:
#                 self.task_train += 1
#             return next_sample
#
# def val_sampler(self, batch_size=4):
#     if self.dataset.tasks >= self.task_val:
#         val_subset = Subset(self.dataset, self.val_indexes[self.task_val])
#         val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
#         next_sample = next(iter(val_loader))
#         if val_loader.__len__() == 0:
#             self.task_val += 1
#         return next_sample
#
# def test_sampler(self, batch_size=4):
#     if self.dataset.tasks >= self.task_test:
#         test_subset = Subset(self.dataset, self.test_indexes[self.task_test])
#         test_loader = DataLoader(test_subset, shuffle=True, batch_size=batch_size)
#         next_sample = next(iter(test_loader))
#         if test_loader.__len__() == 0:
#             self.task_test += 1
#         return next_sample


def extract(nested_list, index):
    temp = []
    for class_ in nested_list:
        for i in index:
            temp.append(class_[i])
    return temp


def sample_dataset_train(train_size, indexes, samples_per_class):
    train_indexes = sorted(
        [y for sub in [indexes[x::samples_per_class] for x in range(0, train_size)] for y in sub])
    train = [train_indexes[i:i + train_size] for i in range(0, len(train_indexes), train_size)]
    random_train = np.random.choice(np.arange(train_size), (2,), replace=False)
    return extract(train, random_train)


def sample_dataset_val(train_size, test_size, indexes, samples_per_class):
    cutoff = train_size + test_size  # 80%

    val_indexes = sorted(
        [y for sub in [indexes[x::samples_per_class] for x in range(train_size, cutoff)] for y in sub])
    val = [val_indexes[i:i + test_size] for i in range(0, len(val_indexes), test_size)]
    random_test = np.random.choice(np.arange(test_size), (2,), replace=True)
    return extract(val, random_test)


def sample_dataset_test(train_size, test_size, indexes, samples_per_class):
    cutoff = train_size + test_size  # 80%

    test_indexes = sorted(
        [y for sub in [indexes[x::samples_per_class] for x in range(cutoff, samples_per_class)] for y in sub])
    test = [test_indexes[i:i + test_size] for i in range(0, len(test_indexes), test_size)]
    random_test = np.random.choice(np.arange(test_size), (2,), replace=True)
    return extract(test, random_test)


class CustomSampler:
    def __init__(self, celeb_dataset, global_labels=False):
        self.dataset = celeb_dataset
        self.samples_per_class = celeb_dataset.samples_per_class
        self.num_tasks = self.dataset.tasks  # 3
        self.number_of_samples = len(self.dataset)
        self.sample_size = 10  # essentially the batch size
        self.train_size = int(np.floor(self.dataset.samples_per_class * .6))
        self.test_val_size = int(np.floor(self.dataset.samples_per_class * .2))
        self.train_indexes, self.val_indexes, self.test_indexes = [], [], []
        if global_labels:
            self.indexes = np.arange(self.dataset.tasks * self.dataset.samples_per_task)
        else:
            task = np.random.randint(self.dataset.tasks)
            self.indexes = np.arange(task * self.dataset.samples_per_task, (task + 1) * self.dataset.samples_per_task)

    def train_sampler(self):
        self.train_indexes = sample_dataset_train(self.train_size, self.indexes, self.samples_per_class)
        train_subset = Subset(self.dataset, self.train_indexes)
        train_loader = DataLoader(train_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(train_loader))
        return next_sample

    def val_sampler(self):
        self.val_indexes = sample_dataset_val(self.train_size, self.test_val_size, self.indexes, self.samples_per_class)
        val_subset = Subset(self.dataset, self.val_indexes)
        val_loader = DataLoader(val_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(val_loader))
        return next_sample

    def test_sampler(self):
        self.test_indexes = sample_dataset_test(self.train_size, self.test_val_size, self.indexes,
                                                self.samples_per_class)
        test_subset = Subset(self.dataset, self.test_indexes)
        test_loader = DataLoader(test_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(test_loader))
        return next_sample


# class CustomBenchmarkSampler:
#     def sample_dataset_train(self, train_size, indexes, samples_per_class):
#         train_indexes = sorted(
#             [y for sub in [indexes[x::samples_per_class] for x in range(0, train_size)] for y in sub])
#         train = [train_indexes[i:i + train_size] for i in range(0, len(train_indexes), train_size)]
#         random_train = np.random.choice(np.arange(train_size), (2,), replace=False)
#         self.train_indexes = extract(train, random_train)
#
#     def sample_dataset_val(self, train_size, test_size, indexes, samples_per_class):
#         cutoff = train_size + test_size  # 80%
#
#         val_indexes = sorted(
#             [y for sub in [indexes[x::samples_per_class] for x in range(train_size, cutoff)] for y in sub])
#         val = [val_indexes[i:i + test_size] for i in range(0, len(val_indexes), test_size)]
#         random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
#         self.val_indexes = extract(val, random_test)
#
#     def sample_dataset_test(self, train_size, test_size, indexes, samples_per_class):
#         cutoff = train_size + test_size  # 8
#
#         test_indexes = sorted(
#             [y for sub in [indexes[x::samples_per_class] for x in range(cutoff, samples_per_class)] for y in sub])
#         test = [test_indexes[i:i + test_size] for i in range(0, len(test_indexes), test_size)]
#         random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
#         self.test_indexes = extract(test, random_test)
#
#     def __init__(self, celeb_dataset):
#         self.dataset = celeb_dataset
#         self.samples_per_class = celeb_dataset.samples_per_class
#         self.num_tasks = self.dataset.tasks  # 3
#         self.number_of_samples = len(self.dataset)
#         self.sample_size = 10  # essentially the batch size
#         self.train_size = int(np.floor(self.samples_per_class * .6))
#         self.test_val_size = int(np.floor(self.samples_per_class * .2))
#         self.indexes = np.arange(self.dataset.tasks * self.dataset.samples_per_task)
#         self.train_indexes, self.val_indexes, self.test_indexes = [], [], []
#
#     def train_sampler(self):
#         self.sample_dataset_train(self.train_size, self.indexes, self.samples_per_class)
#         train_subset = Subset(self.dataset, self.train_indexes)
#         train_loader = DataLoader(train_subset, shuffle=True, batch_size=self.sample_size)
#         next_sample = next(iter(train_loader))
#         return next_sample
#
#     def val_sampler(self):
#         self.sample_dataset_val(self.train_size, self.test_val_size, self.indexes, self.samples_per_class)
#         val_subset = Subset(self.dataset, self.val_indexes)
#         val_loader = DataLoader(val_subset, shuffle=False, batch_size=self.sample_size)
#         next_sample = next(iter(val_loader))
#         return next_sample
#
#     def test_sampler(self):
#         self.sample_dataset_test(self.train_size, self.test_val_size, self.indexes, self.samples_per_class)
#         test_subset = Subset(self.dataset, self.test_indexes)
#         test_loader = DataLoader(test_subset, shuffle=False, batch_size=self.sample_size)
#         next_sample = next(iter(test_loader))
#         return next_sample


if __name__ == '__main__':
    # # tasks = [3, 5]
    # # classes = [5, 10]
    # # batch_size = [10, 15]
    # # for task in tasks:
    # #     for class_ in classes:
    # #         for batch in batch_size:
    # #             dataset = CustomDataset(tasks=task, classes=class_, batch_size=batch)
    # #             dataset_task_splitter(dataset)
    # image_size = 128
    # transformation = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size)
    #     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    #
    # dataset = CustomDataset(tasks=1000, classes=5000, transform=transformation, image_size=image_size)
    # train_sampler = CustomSampler(dataset)
    # print(train_sampler.train_sampler()[1].T)
    #
    # dataset = CustomDataset(tasks=3, classes=15, transform=transformation, image_size=image_size)
    # train_sampler = CustomSampler(dataset)
    # print(train_sampler.train_sampler()[1].T)
    # # train_sampler = CustomBenchmarkSampler(dataset, train_ways=5, train_samples=2, test_ways=5, test_samples=2)
    # # print(train_sampler.train_sampler()[1].T)
    # # print(train_sampler.val_sampler()[1].T)
    # # print(train_sampler.test_sampler()[1].T)
    # # print(train_loaderA)
    # # for task in range(0, dataset.tasks):
    # #     print(len(next(iter(train_loaderA.test_loader(task)))[0]))
    # # print((next(iter(train_loaderA.test_loader(task)))[0]))
    # # dataset_task_splitter(dataset)
    # # print(dataset)
    # # class_number = dataset.__len__()
    # # train_size = int(0.8 * len(dataset))
    # # test_size = len(dataset) - train_size
    # # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # #
    # # train_loader = DataLoader(train_dataset, batch_size=12, shuffle=False)
    # # test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    # # x = next(iter(train_loader))
    # # y = next(iter(test_loader))
    #
    # # dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    # # x = next(iter(dataloader))
    # # print(next(iter(dataloader)))
    #
    # # for imgs, labels in dataloader:
    # #     print("Batch of images has shape: ", imgs.shape)
    # #     print("Batch of labels has shape: ", labels.shape)
    # # dataset, label = create_dataset(shots, ways, meta_batch_size, dataroot, labels_path, image_size=32
    # # batch, dataset, label = batch_loader(5, dataset, class_number)
    image_size = 100
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_accuracy_celeb = 0
    num_tasks = 10
    ways = 500
    shots = 20
    iterations = 1
    batch_size = 16

    dataset = CustomDataset(tasks=1000, classes=5000, transform=transformation, image_size=image_size)
    train_sampler = CustomSampler(dataset, global_labels=True)
    print(train_sampler.test_sampler()[1].T)
    train_sampler = CustomSampler(dataset, global_labels=True)
    print(train_sampler.test_sampler()[1].T)
    train_sampler = CustomSampler(dataset, global_labels=True)
    print(train_sampler.test_sampler()[1].T)

    train_sampler2 = CustomSampler(dataset, global_labels=False)
    print(train_sampler2.test_sampler()[1].T)
    train_sampler2 = CustomSampler(dataset, global_labels=False)
    print(train_sampler2.test_sampler()[1].T)
    train_sampler2 = CustomSampler(dataset, global_labels=False)
    print(train_sampler2.test_sampler()[1].T)
