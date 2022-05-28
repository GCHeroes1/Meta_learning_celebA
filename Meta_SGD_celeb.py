import random
import numpy as np
import torch
import torchvision.transforms as transforms
import learn2learn as l2l
from line_profiler_pycharm import profile
from tqdm import tqdm
from celebA_dataset_creation import CustomDataset, CustomSampler
from torch import nn, optim
from learn2learn.optim.transforms import MetaCurvatureTransform

workers = 4
ngpu = 1
dataroot = r"./CelebA-20220516T115258Z-001/CelebA/Img/img_align_celeba/img_align_celeba"
labels_path = r"./CelebA-20220516T115258Z-001/CelebA/Anno/identity_CelebA.txt"
image_size = 100
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size)
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# N_tasks = 5
# n_classes = 5
# k_samples = 5
# imagesize = 64
# torch.set_default_dtype(torch.float)
import sys


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    # print(predictions, targets)
    # sys.exit()
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, device):
    # batch = batch.float()
    # print(batch)
    data, labels = batch
    data, labels = data.to(device), labels.to(device).squeeze()

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(int(len(adaptation_indices) / 2)) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)  # the validation error is way off
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(tasks, ways, shots, meta_lr=0.003, fast_lr=0.5, meta_batch_size=32, adaptation_steps=1, num_iterations=60000,
         cuda=True, seed=42, global_labels=False):
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
    model = l2l.vision.models.ResNet12(ways, hidden_size=2560)
    # model = l2l.vision.models.ResNet12(ways, hidden_size=2560)
    model.to(device, dtype=torch.float)

    model.to(device)
    metaSGD = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
    metaSGD.to(device)
    opt = optim.Adam(metaSGD.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # dataset = CustomDataset(tasks=3, classes=15, transform=transformation, image_size=image_size)
    dataset = CustomDataset(tasks=tasks, classes=ways, samples_per_class=shots, img_path=dataroot,
                            label_path=labels_path, transform=transformation, image_size=image_size)

    data_plot = []

    for iteration in tqdm(range(num_iterations)):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        for task in range(meta_batch_size):
            # Compute meta-training loss
            sampler = CustomSampler(dataset, global_labels=global_labels)
            learner = metaSGD.clone()
            evaluation_error, evaluation_accuracy = \
                fast_adapt(sampler.train_sampler(), learner, loss, adaptation_steps, device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            sampler = CustomSampler(dataset, global_labels=global_labels)
            learner = metaSGD.clone()
            evaluation_error, evaluation_accuracy = \
                fast_adapt(sampler.val_sampler(), learner, loss, adaptation_steps, device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        # print('\n')
        # print('Iteration', iteration)
        train_err = meta_train_error / meta_batch_size
        train_acc = meta_train_accuracy / meta_batch_size
        val_err = meta_valid_error / meta_batch_size
        val_acc = meta_valid_accuracy / meta_batch_size
        # print('Meta Train Error', train_err)
        # print('Meta Train Accuracy', train_acc)
        # print('Meta Valid Error', val_err)
        # print('Meta Valid Accuracy', val_acc)
        data_plot.append((iteration, train_err, train_acc, val_err, val_acc))
        # print('\n')

        # Average the accumulated gradients and optimize
        for p in metaSGD.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    # dataset = CustomDataset(tasks=tasks, classes=ways, samples_per_class=shots, img_path=dataroot,
    #                         label_path=labels_path, transform=transformation, image_size=image_size)
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        sampler = CustomSampler(dataset, global_labels=global_labels)
        learner = metaSGD.clone()
        evaluation_error, evaluation_accuracy = \
            fast_adapt(sampler.test_sampler(), learner, loss, adaptation_steps, device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    test_err = meta_test_error / meta_batch_size
    test_acc = meta_test_accuracy / meta_batch_size
    print('Meta Test Error', test_err)
    print('Meta Test Accuracy', test_acc)
    return data_plot, test_acc


if __name__ == '__main__':
    # """
    # :param train_ways: number of classes per training batch
    # :param train_samples: number of samples per training batch
    # :param test_ways: number of classes per test/val batch
    # :param test_samples: number of samples per test/val batch
    # :param num_tasks: number of tasks in each dataset
    # """
    # test_accuracy = 0
    # num_tasks = 5
    # ways = 5
    # shots = 1
    # for i in range(10):
    #     print('Iteration', i + 1)
    #     test_accuracy += main(tasks=num_tasks, ways=ways * num_tasks, meta_batch_size=16,
    #                           shots=shots, num_iterations=10, global_labels=False)
    # print(test_accuracy / 10)
    import csv

    num_tasks = 10
    ways = 500
    shots = 5
    iterations = 30
    batch_size = 256
    global_labels = False

    data_plot, accuracy = main(tasks=10,
                               ways=500,
                               shots=5,
                               num_iterations=30,
                               meta_batch_size=256,
                               global_labels=False)
    with open(
            f"./results/Meta-SGD/celebA/{num_tasks}_{ways}_{shots}_"
            f"{iterations}_{batch_size}_{str(global_labels)}",
            "w") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data_plot)
