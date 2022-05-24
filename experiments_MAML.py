import MAML_celeb
import MAML_omniglot
import sys

if __name__ == '__main__':

    """
    :param train_ways: number of classes per training batch
    :param train_samples: number of samples per training batch
    :param test_ways: number of classes per test/val batch
    :param test_samples: number of samples per test/val batch
    :param num_tasks: number of tasks in each dataset
    """
    test_accuracy_celeb = 0
    num_tasks = 5
    ways_num_classes_per_task = 5
    shots_num_samples_per_class = 1
    iterations = 10
    # meta_lrs = [0.006, 0.005, 0.007, 0.1]
    # fast_lrs = [0.4, 0.5, 0.6, 0.7, 0.8]
    meta_lrs = [0.006, 0.005, 0.007]
    fast_lrs = [0.65, 0.7, 0.75]
    # (0.09750000145286322, 0.005, 0.7)
    # omniglot accuracy 0.10416667349636555

    # (0.04500000067055225, 0.006, 0.75)
    accuracystats = []
    for meta_lr in meta_lrs:
        for fast_lr in fast_lrs:
            test_accuracy_celeb = 0
            for i in range(iterations):
                print('Iteration', i + 1)
                test_accuracy_celeb += MAML_celeb.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
                                                       meta_batch_size=16, meta_lr=meta_lr, fast_lr=fast_lr,
                                                       shots=shots_num_samples_per_class, num_iterations=iterations)
            print("meta_lr: ", meta_lr, "fast_lr: ", fast_lr, "celeb accuracy: ", test_accuracy_celeb / iterations)
            accuracystats.append((test_accuracy_celeb / iterations, meta_lr, fast_lr))
    maximum_acc = max(accuracystats, key=lambda i: i[0])
    print(maximum_acc)

    # test_accuracy_omniglot = 0
    # for i in range(iterations):
    #     print('Iteration', i + 1)
    #     test_accuracy_omniglot += MAML_omniglot.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
    #                                                  meta_batch_size=16,
    #                                                  shots=shots_num_samples_per_class, num_iterations=iterations)
    #
    # # print("celeb accuracy ", test_accuracy_celeb / 10)
    #
    # print("omniglot accuracy ", test_accuracy_omniglot / 10)
    #
    # # celeb accuracy 0.07312500132247805
    # # omniglot accuracy 0.11666667461395264
