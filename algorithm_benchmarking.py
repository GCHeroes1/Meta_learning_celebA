"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
import learn2learn as l2l
from CifarCNN import CifarCNN
from learn2learn.optim.transforms import MetaCurvatureTransform


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data = data[adaptation_indices]
    adaptation_labels = labels[adaptation_indices]
    evaluation_data = data[evaluation_indices]
    evaluation_labels = labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(model, algorithm, taskset, tasks, ways=5, shots=1, meta_lr=0.003, meta_batch_size=32,
         adaptation_steps=1,
         num_iterations=10,
         cuda=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    # Load train/validation/test tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(taskset,
                                                  train_ways=ways,
                                                  train_samples=2 * shots,
                                                  test_ways=ways,
                                                  test_samples=2 * shots,
                                                  num_tasks=tasks,
                                                  root=f'./data/{taskset}',
                                                  )

    # # Create model
    # if taskset == "omniglot":
    #     model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    # elif taskset == "omniglotCNN":
    #     model = l2l.vision.models.OmniglotCNN(ways)
    # elif taskset == "mini-imagenet":
    #     model = l2l.vision.models.MiniImagenetCNN(ways)
    # elif taskset == "fc100":
    #     model = CifarCNN(output_size=ways)
    # elif taskset == "fc100_WRN28":
    #     model = l2l.vision.models.WRN28(output_size=ways)
    # elif taskset == "celebA":
    #     model = l2l.vision.models.ResNet12(ways, hidden_size=2560)
    # else:
    #     model = l2l.vision.models.ResNet12(ways)

    model.to(device)
    # metaSGD = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
    algorithm.to(device)
    opt = optim.Adam(algorithm.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    data_plot = []

    for iteration in tqdm(range(num_iterations)):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = algorithm.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = \
                fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = algorithm.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = \
                fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device)
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
        for p in algorithm.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = algorithm.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = \
            fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    test_err = meta_test_error / meta_batch_size
    test_acc = meta_test_accuracy / meta_batch_size
    print('Meta Test Error', test_err)
    print('Meta Test Accuracy', test_acc)
    return data_plot, test_acc


if __name__ == '__main__':
    # tasksets = ["omniglot", "mini-imagenet", "fc100"]
    tasksets = ["fc100"]
    for taskset in tasksets:
        if taskset == "omniglot":
            main(taskset="omniglot", tasks=10,
                 ways=100,
                 shots=1,
                 num_iterations=1500,
                 meta_batch_size=32)
            # 3hr11 for 1500 epoch
        elif taskset == "mini-imagenet":
            main(taskset="mini-imagenet", tasks=10,
                 ways=15,
                 shots=5,
                 num_iterations=1500,
                 meta_batch_size=32)
            # 2hr2 for 1500 epoch
        elif taskset == "fc100":
            main(taskset="fc100", tasks=1000,
                 ways=15,
                 shots=1,
                 num_iterations=2000,
                 meta_batch_size=32)
    # num_tasks = 10
    # ways = 100
    # shots = 1
    # iterations = 1500
    # batch_size = 32
    #
    # main(taskset=taskset, tasks=num_tasks, ways=ways,
    #      meta_batch_size=batch_size, shots=shots,
    #      num_iterations=iterations)
