import MAML_celeb
import MAML_omniglot
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv

if __name__ == '__main__':
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    """
    :param train_ways: number of classes per training batch
    :param train_samples: number of samples per training batch
    :param test_ways: number of classes per test/val batch
    :param test_samples: number of samples per test/val batch
    :param num_tasks: number of tasks in each dataset
    """
    test_accuracy_celeb = 0
    num_tasks = 10
    ways_num_classes_per_task = 500
    shots_num_samples_per_class = 5
    iterations = 10
    batch_size = 512
    global_labels = True
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
    data_plot, accuracy = MAML_celeb.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
                                          meta_batch_size=batch_size, shots=shots_num_samples_per_class,
                                          num_iterations=iterations, global_labels=global_labels)
    with open(
            f"./results/{num_tasks}_{ways_num_classes_per_task}_{shots_num_samples_per_class}_"
            f"{iterations}_{batch_size}_{str(global_labels)}",
            "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data_plot)
        # csvWriter.writerows(accuracy)
    print("final accuracy", accuracy)

    iteration_list = [n[0] for n in data_plot]
    train_err = [m[1] for m in data_plot]
    train_acc = [z[2] for z in data_plot]
    val_err = [x[3] for x in data_plot]
    val_acc = [c[4] for c in data_plot]

    plt.plot(iteration_list, train_err, color='blue', label='Training Error')
    plt.plot(iteration_list, val_err, color='orange', label='Validation Error')
    plt.title(
        f"train and val error for {num_tasks} tasks with {ways_num_classes_per_task} classes each, "
        f"\nglobal labels are {global_labels}, batch size is {batch_size}", horizontalalignment='center')
    plt.xlabel("iteration")
    plt.ylabel("training and val error")
    plt.legend()
    plt.savefig(
        f'./plots/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_err_{batch_size}_{iterations}_{global_labels}.png')
    plt.show()
    plt.clf()

    plt.plot(iteration_list, train_acc, color='navy', label='Training Accuracy')
    plt.plot(iteration_list, val_acc, color='goldenrod', label='Validation Error')
    plt.title(
        f"train and val acc for {num_tasks} tasks with {ways_num_classes_per_task} classes each, "
        f"\nglobal labels are {global_labels}, batch size is {batch_size}", horizontalalignment='center')
    plt.xlabel("iteration")
    plt.ylabel("training and val acc")
    plt.legend()
    plt.savefig(
        f'./plots/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_acc_{batch_size}_{iterations}_{global_labels}.png')
    plt.show()
    plt.clf()
