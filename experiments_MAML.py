import MAML_celeb
import MAML_omniglot
import sys
from tqdm import tqdm

if __name__ == '__main__':
    """
    :param train_ways: number of classes per training batch
    :param train_samples: number of samples per training batch
    :param test_ways: number of classes per test/val batch
    :param test_samples: number of samples per test/val batch
    :param num_tasks: number of tasks in each dataset
    """
    test_accuracy_celeb = 0
    num_tasks = 10
    ways_num_classes_per_task = 10
    shots_num_samples_per_class = 10
    iterations = 10
    batch_size = shots_num_samples_per_class * ways_num_classes_per_task
    # # meta_lrs = [0.006, 0.005, 0.007]
    # # fast_lrs = [0.4, 0.5, 0.6]
    # # meta_lrs = [0.006, 0.005, 0.007]
    # # fast_lrs = [0.65, 0.7, 0.75]
    # meta_lrs = [0.005]
    # fast_lrs = [0.7]
    # # (0.09750000145286322, 0.005, 0.7)
    # # omniglot accuracy 0.10416667349636555
    #
    # # (0.04500000067055225, 0.006, 0.75)
    #
    # # (0.03833333542570472, 0.005, 0.7, 3, 5, 10) <- benchmark, celeb
    # # (0.05000000074505806, 0.005, 0.7, 3, 5, 10) <- no global labels
    # # omniglot -> 0.1208333414979279
    # accuracystats = []
    # for meta_lr in meta_lrs:
    #     for fast_lr in fast_lrs:
    #         test_accuracy_celeb = 0
    #         for i in tqdm(range(5)):
    #             # print('Iteration', i + 1)
    #             test_accuracy_celeb += MAML_celeb.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
    #                                                    meta_batch_size=batch_size, meta_lr=meta_lr, fast_lr=fast_lr,
    #                                                    shots=shots_num_samples_per_class, num_iterations=iterations)
    #         print("meta_lr: ", meta_lr, "fast_lr: ", fast_lr, "celeb accuracy: ", test_accuracy_celeb / 5)
    #         accuracystats.append(
    #             (test_accuracy_celeb / 5, meta_lr, fast_lr, num_tasks, ways_num_classes_per_task, iterations))
    # maximum_acc = max(accuracystats, key=lambda i: i[0])
    # print(maximum_acc)
    #
    # test_accuracy_omniglot = 0
    # for i in tqdm(range(1)):
    #     # print('Iteration', i + 1)
    #     test_accuracy_omniglot += MAML_omniglot.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
    #                                                  meta_batch_size=100,
    #                                                  shots=shots_num_samples_per_class, num_iterations=1000)

    # # print("celeb accuracy ", test_accuracy_celeb / 10)
    # #
    # print("omniglot accuracy ", test_accuracy_omniglot / 1)
    # #
    # # # celeb accuracy 0.07312500132247805
    # # # omniglot accuracy 0.11666667461395264
    accuracy = 0
    for i in range(1):
        accuracy += MAML_celeb.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
                                    meta_batch_size=50,
                                    shots=shots_num_samples_per_class, num_iterations=2000)
    print(accuracy / 1)
    # 0.08400000125169754 with 3 tasks, 5 samples each
    # 0.29 was the best it did, even when trying to force overfitting
