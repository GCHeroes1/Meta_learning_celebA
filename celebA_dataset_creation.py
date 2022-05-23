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


# if torch.cuda.is_available():
#     device = torch.device('cuda')


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
            classes = np.arange(0, self.classes)
            class_map = {}
            keys_to_be_deleted = []
            for y, (class_remapped, (key, val)) in enumerate(zip(classes, existing_mapping.items())):
                class_map[y + i * self.classes] = val
                for each in val:
                    self.data.append([dataroot + os.path.sep + each, y + i * self.classes])
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

        self.img_path = img_path
        self.label_path = label_path
        # this creates a dict of lists, each key is the class, each value is list of files corresponding to that class
        self.class_map = self.create_class_map(self.samples_per_class)
        # trims the lists down
        self.class_map = {key: val for key, val in self.class_map.items() if len(val) >= self.samples_per_class}

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


class CustomLoader:
    def segment_dataset(self):
        for task in range(self.dataset.tasks):
            indexes = np.arange(task * self.number_of_samples, (task + 1) * self.number_of_samples)
            self.train_indexes.append(np.random.choice(indexes, self.train_size, replace=False))
            indexes = np.delete(indexes, np.where(np.isin(indexes, self.train_indexes)))
            self.val_indexes.append(np.random.choice(indexes, self.val_size, replace=False))
            indexes = np.delete(indexes, np.where(np.isin(indexes, self.val_indexes)))
            self.test_indexes.append(np.random.choice(indexes, self.test_size, replace=False))
            # indexes = np.delete(indexes, np.where(np.isin(indexes, self.test_indexes)))

    def __init__(self, celeb_dataset):
        self.dataset = celeb_dataset
        self.number_of_samples = self.dataset.classes * self.dataset.samples_per_class
        self.test_size = int(np.floor(self.number_of_samples / self.dataset.class_size))
        self.train_size = int((self.number_of_samples - self.test_size) * .8)
        self.val_size = self.number_of_samples - self.test_size - self.train_size
        self.train_indexes = []
        self.val_indexes = []
        self.test_indexes = []
        self.segment_dataset()
        self.task_train = 0
        self.task_val = 0
        self.task_test = 0

    def train_loader(self, current_task, batch_size=4):
        train_subset = Subset(self.dataset, self.train_indexes[current_task])
        train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
        return train_loader

    def val_loader(self, current_task, batch_size=4):
        val_subset = Subset(self.dataset, self.val_indexes[current_task])
        val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
        return val_loader

    def test_loader(self, current_task, batch_size=4):
        test_subset = Subset(self.dataset, self.test_indexes[current_task])
        test_loader = DataLoader(test_subset, shuffle=False, batch_size=batch_size)
        return test_loader

    def train_sampler(self, batch_size=4):
        if self.dataset.tasks >= self.task_train:
            train_subset = Subset(self.dataset, self.train_indexes[self.task_train])
            train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
            next_sample = next(iter(train_loader))
            if train_loader.__len__() == 0:
                self.task_train += 1
            return next_sample

    def val_sampler(self, batch_size=4):
        if self.dataset.tasks >= self.task_val:
            val_subset = Subset(self.dataset, self.val_indexes[self.task_val])
            val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
            next_sample = next(iter(val_loader))
            if val_loader.__len__() == 0:
                self.task_val += 1
            return next_sample

    def test_sampler(self, batch_size=4):
        if self.dataset.tasks >= self.task_test:
            test_subset = Subset(self.dataset, self.test_indexes[self.task_test])
            test_loader = DataLoader(test_subset, shuffle=True, batch_size=batch_size)
            next_sample = next(iter(test_loader))
            if test_loader.__len__() == 0:
                self.task_test += 1
            return next_sample


class CustomSampler:
    def extract(self, nested_list, index):
        temp = []
        for class_ in nested_list:
            for i in index:
                temp.append(class_[i])
        return temp

    def sample_dataset_train(self, task, train_size=6):
        indexes = np.arange(task * self.number_of_samples, (task + 1) * self.number_of_samples)

        train_indexes = sorted(
            [y for sub in [indexes[x::self.sample_size] for x in range(0, train_size)] for y in sub])
        train = [train_indexes[i:i + train_size] for i in range(0, len(train_indexes), train_size)]
        random_train = np.random.choice(np.arange(train_size), (2,), replace=False)
        self.train_indexes = self.extract(train, random_train)

    def sample_dataset_val(self, task, train_size=6, test_size=2):
        indexes = np.arange(task * self.number_of_samples, (task + 1) * self.number_of_samples)
        val_cutoff = train_size + test_size  # 8

        val_indexes = sorted(
            [y for sub in [indexes[x::self.sample_size] for x in range(train_size, val_cutoff)] for y in sub])
        val = [val_indexes[i:i + test_size] for i in range(0, len(val_indexes), test_size)]
        random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
        self.val_indexes = self.extract(val, random_test)

    def sample_dataset_test(self, task, train_size=6, test_size=2):
        indexes = np.arange(task * self.number_of_samples, (task + 1) * self.number_of_samples)
        val_cutoff = train_size + test_size  # 8

        test_indexes = sorted(
            [y for sub in [indexes[x::self.sample_size] for x in range(val_cutoff, self.sample_size)] for y in sub])
        test = [test_indexes[i:i + test_size] for i in range(0, len(test_indexes), test_size)]
        random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
        self.test_indexes = self.extract(test, random_test)

    def __init__(self, celeb_dataset):
        """
        :param train_ways: number of classes per training batch
        :param train_samples: number of samples per training batch
        :param test_ways: number of classes per test/val batch
        :param test_samples: number of samples per test/val batch
        :param num_tasks: number of tasks in each dataset
        """
        self.dataset = celeb_dataset
        self.num_tasks = self.dataset.tasks  # 3
        # this is the number of samples in the dataset, 150
        # self.number_of_samples = self.dataset.classes * self.dataset.samples_per_class
        self.number_of_samples = len(self.dataset)
        self.sample_size = celeb_dataset.classes_per_task * 2  # 10 or 30, depends on number of classes per task
        self.train_size = int(np.floor(self.sample_size * .6))
        self.test_val_size = int(np.floor(self.sample_size * .2))
        # self.test_val_size = test_ways * test_samples  # this is (shots * 2) * ways
        # self.train_size = train_ways * train_samples  # this is also (shots *2) * ways
        self.train_indexes = []
        self.val_indexes = []
        self.test_indexes = []
        self.task_train = 0
        self.task_val = 0
        self.task_test = 0

    def train_sampler(self):
        self.sample_dataset_train(np.random.randint(self.dataset.tasks), self.train_size)
        train_subset = Subset(self.dataset, self.train_indexes)
        train_loader = DataLoader(train_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(train_loader))
        return next_sample

    def val_sampler(self):
        self.sample_dataset_val(np.random.randint(self.dataset.tasks), self.train_size, self.test_val_size)
        val_subset = Subset(self.dataset, self.val_indexes)
        val_loader = DataLoader(val_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(val_loader))
        return next_sample

    def test_sampler(self):
        self.sample_dataset_test(np.random.randint(self.dataset.tasks), self.train_size, self.test_val_size)
        test_subset = Subset(self.dataset, self.test_indexes)
        test_loader = DataLoader(test_subset, shuffle=True, batch_size=self.sample_size)
        next_sample = next(iter(test_loader))
        return next_sample


class CustomBenchmarkSampler:

    def extract(self, nested_list, index):
        temp = []
        for class_ in nested_list:
            for i in index:
                temp.append(class_[i])
        return temp

    def sample_dataset(self, type, train_size=6, test_size=2):
        indexes = np.arange(len(self.dataset))
        # indexes = np.arange(0 * self.number_of_samples, (0 + 1) * self.number_of_samples)
        val_cutoff = train_size + test_size  # 8

        if type == "train":
            train_indexes = sorted(
                [y for sub in [indexes[x::self.train_size] for x in range(0, train_size)] for y in sub])
            train = [train_indexes[i:i + train_size] for i in range(0, len(train_indexes), train_size)]
            random_train = np.random.choice(np.arange(train_size), (2,), replace=False)
            self.train_indexes = self.extract(train, random_train)
            return
        if type == "val":
            val_indexes = sorted(
                [y for sub in [indexes[x::self.train_size] for x in range(train_size, val_cutoff)] for y in sub])
            val = [val_indexes[i:i + test_size] for i in range(0, len(val_indexes), test_size)]
            random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
            self.val_indexes = self.extract(val, random_test)
            return
        if type == "test":
            test_indexes = sorted(
                [y for sub in [indexes[x::self.train_size] for x in range(val_cutoff, self.train_size)] for y in sub])
            test = [test_indexes[i:i + test_size] for i in range(0, len(test_indexes), test_size)]
            random_test = np.random.choice(np.arange(test_size), (2,), replace=False)
            self.test_indexes = self.extract(test, random_test)
            return

    def __init__(self, celeb_dataset, train_ways, train_samples, test_ways, test_samples):
        """
        :param train_ways: number of classes per training batch
        :param train_samples: number of samples per training batch
        :param test_ways: number of classes per test/val batch
        :param test_samples: number of samples per test/val batch
        :param num_tasks: number of tasks in each dataset
        """
        # you need to make sure that theres 2 samples of each class in each sample you provide, so 15 classes must be a 30 sized sample.
        self.dataset = celeb_dataset
        self.num_tasks = self.dataset.tasks
        self.number_of_samples = self.dataset.classes * self.dataset.samples_per_class  # this is the number of samples in each task
        self.test_val_size = test_ways * test_samples  # this is (shots * 2) * ways
        self.train_size = train_ways * train_samples  # this is also (shots *2) * ways
        self.train_indexes = []
        self.val_indexes = []
        self.test_indexes = []
        self.task_train = 0
        self.task_val = 0
        self.task_test = 0

    def train_sampler(self):
        self.sample_dataset("train", self.train_size - 2 * 2, 2)
        train_subset = Subset(self.dataset, self.train_indexes)
        train_loader = DataLoader(train_subset, shuffle=True, batch_size=self.train_size)
        next_sample = next(iter(train_loader))
        return next_sample

    def val_sampler(self):
        self.sample_dataset("val", self.train_size - 2 * 2, 2)
        val_subset = Subset(self.dataset, self.val_indexes)
        val_loader = DataLoader(val_subset, shuffle=True, batch_size=self.test_val_size)
        next_sample = next(iter(val_loader))
        return next_sample

    def test_sampler(self):
        self.sample_dataset("test", self.train_size - 2 * 2, 2)
        test_subset = Subset(self.dataset, self.test_indexes)
        test_loader = DataLoader(test_subset, shuffle=True, batch_size=self.test_val_size)
        next_sample = next(iter(test_loader))
        return next_sample


# # testing subsampling dataset and creating loaders
# def dataset_task_splitter(dataset: CustomDataset):
#     number_of_samples = dataset.classes * dataset.samples_per_class
#     test_size = int(np.floor(number_of_samples / (dataset.batch_size)))
#     train_size = int((number_of_samples - test_size) * .8)
#     val_size = number_of_samples - test_size - train_size
#
#     # indexes = np.arange(task * number_of_samples, (task + 1) * number_of_samples)
#     # train_indexes = np.random.choice(indexes, train_size, replace=False)
#     # indexes = np.delete(indexes, np.where(np.isin(indexes, train_indexes)))
#     # val_indexes = np.random.choice(indexes, val_size, replace=False)
#     # indexes = np.delete(indexes, np.where(np.isin(indexes, val_indexes)))
#     # test_indexes = np.random.choice(indexes, test_size, replace=False)
#     # indexes = np.delete(indexes, np.where(np.isin(indexes, test_indexes)))
#     # assert (len(indexes) == 0)
#
#     train_indexes, val_indexes, test_indexes = [], [], []
#     for task in range(dataset.tasks):
#         indexes = np.arange(task * number_of_samples, (task + 1) * number_of_samples)
#         train_indexes.append(np.random.choice(indexes, train_size, replace=False))
#         indexes = np.delete(indexes, np.where(np.isin(indexes, train_indexes)))
#         val_indexes.append(np.random.choice(indexes, val_size, replace=False))
#         indexes = np.delete(indexes, np.where(np.isin(indexes, val_indexes)))
#         test_indexes.append(np.random.choice(indexes, test_size, replace=False))
#         indexes = np.delete(indexes, np.where(np.isin(indexes, test_indexes)))
#         assert (len(indexes) == 0)
#     # print(train_indexes, val_indexes, test_indexes)
#
#     # train_subset_A = Subset(dataset, train_indexes[0])
#     # train_subset_B = Subset(dataset, train_indexes[1])
#     # train_subset_C = Subset(dataset, train_indexes[2])
#     #
#     # train_loader_subset_A = DataLoader(train_subset_A, shuffle=True, batch_size=8)
#     # train_loader_subset_B = DataLoader(train_subset_B, shuffle=True, batch_size=8)
#     # train_loader_subset_C = DataLoader(train_subset_C, shuffle=True, batch_size=8)
#     # imgA, labA = next(iter(train_loader_subset_A))
#     # imgB, labB = next(iter(train_loader_subset_B))
#     # imgC, labC = next(iter(train_loader_subset_C))
#     # print("test")
#
#     # for training_subset in train_indexes:
#     #     train_subset = Subset(dataset, training_subset)
#     #     training_loader = DataLoader(train_subset, shuffle=True, batch_size=32)
#     #     for img, lab in training_loader:
#     #         print(lab)
#
#     for (train_subset, val_subset, test_subset) in zip(train_indexes, val_indexes, test_indexes):
#         train_subset = Subset(dataset, train_subset)
#         val_subset = Subset(dataset, val_subset)
#         test_subset = Subset(dataset, test_subset)
#         training_loader = DataLoader(train_subset, shuffle=True, batch_size=32)
#         val_loader = DataLoader(val_subset, shuffle=True, batch_size=32)
#         test_loader = DataLoader(test_subset, shuffle=True, batch_size=32)
#         # for img, lab in training_loader:
#         #     print(len(lab))
#         # for img, lab in val_loader:
#         #     print(len(lab))
#         # for img, lab in test_loader:
#         #     print(len(lab))


if __name__ == '__main__':
    # tasks = [3, 5]
    # classes = [5, 10]
    # batch_size = [10, 15]
    # for task in tasks:
    #     for class_ in classes:
    #         for batch in batch_size:
    #             dataset = CustomDataset(tasks=task, classes=class_, batch_size=batch)
    #             dataset_task_splitter(dataset)
    image_size = 32
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CustomDataset(tasks=1, classes=15, transform=transformation, image_size=image_size)
    # train_loaderA = CustomLoader(dataset)
    train_sampler = CustomSampler(dataset)
    # for i in range(100):
    print(train_sampler.train_sampler()[1].T)

    train_sampler = CustomBenchmarkSampler(dataset, train_ways=5, train_samples=2, test_ways=5, test_samples=2)
    # for i in range(100):
    print(train_sampler.train_sampler()[1].T)
    # print(train_sampler.val_sampler()[1].T)
    # print(train_sampler.test_sampler()[1].T)
    # print(train_loaderA)
    # for task in range(0, dataset.tasks):
    #     print(len(next(iter(train_loaderA.test_loader(task)))[0]))
    # print((next(iter(train_loaderA.test_loader(task)))[0]))
    # dataset_task_splitter(dataset)
    # print(dataset)
    # class_number = dataset.__len__()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=12, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    # x = next(iter(train_loader))
    # y = next(iter(test_loader))

    # dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    # x = next(iter(dataloader))
    # print(next(iter(dataloader)))

    # for imgs, labels in dataloader:
    #     print("Batch of images has shape: ", imgs.shape)
    #     print("Batch of labels has shape: ", labels.shape)
    # dataset, label = create_dataset(shots, ways, meta_batch_size, dataroot, labels_path, image_size=32
    # batch, dataset, label = batch_loader(5, dataset, class_number)
