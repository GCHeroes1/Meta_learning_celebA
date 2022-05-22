import random
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
import learn2learn as l2l

from torch.utils.data import Dataset, DataLoader, Subset
from dataset_construction import CustomImageDataset, MyDataset
from torch import nn, optim

workers = 4
ngpu = 1
dataroot = r"C:\Users\Rajesh\Documents\Pycharm Projects\Y4_COMP0138\Datasets\celebA\CelebA-20220516T115258Z-001\CelebA\Img\img_align_celeba"
image_folder_name = "\img_align_celeba"
labels_path = r"C:\Users\Rajesh\Documents\Pycharm Projects\Y4_COMP0138\Datasets\celebA\CelebA-20220516T115258Z-001\CelebA\Anno\identity_CelebA.txt"
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
N_tasks = 5
n_classes = 5
k_samples = 5
imagesize = 64
torch.set_default_dtype(torch.float)


def create_dataset(N, n, k, dataroot, labelspath, image_size=64):
    transformation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    new_dataset = CustomImageDataset(labelspath, dataroot + image_folder_name, transform=transformation)
    df = new_dataset.img_labels
    new_dataset.__getitem__(1)

    min_df = df.groupby('y').filter(lambda x: len(x) >= k)  # only consider people with >= k samples
    min_classes = min_df["y"].nunique()  # only consider when we have at least n*k classes
    min_data = len(min_df)

    if min_data < N * n * k:
        print("Not enough samples available")
        return None, None
    if min_classes < N * n:
        print("Not enough classes possible")
        return None, None
    # unique_labels = the labels for the samples which fit the criteria required (people with >= k samples)
    unique_labels, counts = np.unique(min_df.y.ravel(), return_counts=True)
    return new_dataset, unique_labels
    return new_dataset, np.arange(0, len(unique_labels - 1))


# this is a batch for a **single class**
def batch_loader(k, dataset, unique_labels):
    label = np.random.choice(unique_labels)
    # label is the class ID, its the ID
    unique_labels = np.delete(unique_labels, np.where(unique_labels == label))
    train_idx = np.where((dataset.img_labels == label))[0]
    # train_idx = the indexes for the images
    train_subset = Subset(dataset, train_idx)
    if len(np.unique(train_idx)) != len(train_idx):  # it should never hit this, duplicate image check
        print(f"why was there a duplicate for {train_idx}?")
        print(np.unique(train_idx))
    train_loader_subset = DataLoader(train_subset, shuffle=True, batch_size=k)  # get batches of the same person
    return next(iter(train_loader_subset)), dataset, unique_labels
    # the batch is returned, as well as the entire dataset and the labels for the samples


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    # print(targets)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    # batch = batch.float()
    # print(batch)
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # this data is correct, it will be k number of images, the label is correct in that, its the actual label corresponding to that person in the original dataset

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        # print(type(adaptation_data.data))
        # print(type(adaptation_labels.data))
        # test = torch.rand(adaptation_labels.shape) * 100
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    """
    **Arguments**
    * **output_size** (int) - The dimensionality of the output (eg, number of classes).
    * **hidden_size** (list, *optional*, default=640) - Size of the embedding once features are extracted.
        (640 is for mini-ImageNet; used for the classifier layer)
    * **avg_pool** (bool, *optional*, default=True) - Set to False for the 16k-dim embeddings of Lee et al, 2019.
    * **wider** (bool, *optional*, default=True) - True uses (64, 160, 320, 640) filters akin to Lee et al, 2019.
        False uses (64, 128, 256, 512) filters, akin to Oreshkin et al, 2018.
    * **embedding_dropout** (float, *optional*, default=0.0) - Dropout rate on the flattened embedding layer.
    * **dropblock_dropout** (float, *optional*, default=0.1) - Dropout rate for the residual layers.
    * **dropblock_size** (int, *optional*, default=5) - Size of drop blocks.
    """
    model = l2l.vision.models.ResNet12(ways, dropblock_dropout=0, avg_pool=False, hidden_size=2560)
    model.to(device, dtype=torch.float)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        dataset, label = create_dataset(shots, ways, meta_batch_size, dataroot, labels_path, image_size=32)
        # print(dataset)
        # dataset = l2l.data.MetaDataset(dataset)
        # taskset = MyDataset(dataset, unique)
        # print(taskset)
        # print("test")
        # print(max(dataset["img_labels"]))
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch, dataset, label = batch_loader(k_samples, dataset, label)
            # print("1, ", batch)
            # batch = batch.float()
            # batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            # batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    dataset, unique = create_dataset(shots, ways, meta_batch_size, dataroot, labels_path, image_size=32)
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch, dataset, unique = batch_loader(k_samples, dataset, unique)
        batch = batch.float()
        # batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    # print("test")
    N_tasks = 1  # number of tasks
    n_classes = 5  # classes per tasks = number of ways
    k_samples = 30  # batch size essentially
    # dataset, unique = create_dataset(N_tasks, n_classes, k_samples, dataroot, labels_path, image_size=64)
    # if dataset is not None:
    #     for t in range(N_tasks):
    #         for c in range(n_classes):
    #             loader, dataset, unique = batch_loader(k_samples, dataset, unique)
    #             print(loader[1])
    main(ways=n_classes, shots=N_tasks, meta_batch_size=k_samples, num_iterations=10)
