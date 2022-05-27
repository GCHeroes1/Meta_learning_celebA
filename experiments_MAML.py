import MAML_celeb
import MAML_benchmarking
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv


def celebA_MAML(num_tasks, ways_num_classes_per_task, shots_num_samples_per_class, iterations, batch_size,
                global_labels):
    data_plot, accuracy = MAML_celeb.main(tasks=num_tasks, ways=ways_num_classes_per_task * num_tasks,
                                          meta_batch_size=batch_size, shots=shots_num_samples_per_class,
                                          num_iterations=iterations, global_labels=global_labels)
    with open(
            f"./results/MAML/celebA/{num_tasks}_{ways_num_classes_per_task}_{shots_num_samples_per_class}_"
            f"{iterations}_{batch_size}_{str(global_labels)}",
            "w") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data_plot)
    print("final accuracy for celebA", accuracy)

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
        f'./plots/MAML/celebA/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_err_{batch_size}_'
        f'{iterations}_{global_labels}.png')
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
        f'./plots/MAML/celebA/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_acc_{batch_size}_'
        f'{iterations}_{global_labels}.png')
    plt.show()
    plt.clf()


def tasksets_MAML(taskset, num_tasks, ways_num_classes_per_task, shots_num_samples_per_class, iterations, batch_size):
    data_plot, accuracy = MAML_benchmarking.main(taskset=taskset, tasks=num_tasks, ways=ways_num_classes_per_task,
                                                 meta_batch_size=batch_size, shots=shots_num_samples_per_class,
                                                 num_iterations=iterations)

    with open(
            f"./results/MAML/{taskset}/{num_tasks}_{ways_num_classes_per_task}_{shots_num_samples_per_class}_"
            f"{iterations}_{batch_size}",
            "w") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data_plot)
    print(f"final accuracy for {taskset}", accuracy)

    iteration_list = [n[0] for n in data_plot]
    train_err = [m[1] for m in data_plot]
    train_acc = [z[2] for z in data_plot]
    val_err = [x[3] for x in data_plot]
    val_acc = [c[4] for c in data_plot]

    plt.plot(iteration_list, train_err, color='blue', label='Training Error')
    plt.plot(iteration_list, val_err, color='orange', label='Validation Error')
    plt.title(
        f"the {taskset} dataset, train and val error for {num_tasks} tasks with {ways_num_classes_per_task} classes "
        f"\neach, batch size is {batch_size}, over {iterations}", horizontalalignment='center')
    plt.xlabel("iteration")
    plt.ylabel("training and val error")
    plt.legend()
    plt.savefig(
        f'./plots/MAML/{taskset}/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_err_{batch_size}_'
        f'{iterations}.png')
    plt.show()
    plt.clf()

    plt.plot(iteration_list, train_acc, color='navy', label='Training Accuracy')
    plt.plot(iteration_list, val_acc, color='goldenrod', label='Validation Error')
    plt.title(
        f"the {taskset} dataset, train and val acc for {num_tasks} tasks with {ways_num_classes_per_task} classes "
        f"\neach, batch size is {batch_size}, over {iterations}", horizontalalignment='center')
    plt.xlabel("iteration")
    plt.ylabel("training and val acc")
    plt.legend()
    plt.savefig(
        f'./plots/MAML/{taskset}/tasks_{str(num_tasks)}_classes_per_{str(ways_num_classes_per_task)}_acc_{batch_size}_'
        f'{iterations}.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    import sys

    algorithms = ["MAML", "GBML", "Meta-SGD"]
    tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    for algorithm in algorithms:
        for taskset in tasksets:

            if not os.path.exists(f'./plots/{algorithm}/{taskset}'):
                os.makedirs(f'./plots/{algorithm}/{taskset}')
            if not os.path.exists(f'./results/{algorithm}/{taskset}'):
                os.makedirs(f'./results/{algorithm}/{taskset}')
    sys.exit()
    # test_accuracy_celeb = 0
    # num_tasks = 10
    # ways_num_classes_per_task = 500
    # shots_num_samples_per_class = 1
    # iterations = 10
    # batch_size = 64
    # global_labels = True

    # remember youre only doing 500 classes because it wont fit on your GPU
    # celebA_MAML(num_tasks=10,
    #             ways_num_classes_per_task=500,
    #             shots_num_samples_per_class=5,
    #             iterations=20,
    #             batch_size=256,
    #             global_labels=True)

    # tasksets = ["omniglot", "mini-imagenet", "fc100"]
    tasksets = ["mini-imagenet"]
    for taskset in tasksets:
        if taskset == "omniglot":
            tasksets_MAML(taskset="omniglot", num_tasks=10,
                          ways_num_classes_per_task=100,
                          shots_num_samples_per_class=1,
                          iterations=1500,
                          batch_size=32)
            # 3hr11 for 1500 epoch
        elif taskset == "mini-imagenet":
            tasksets_MAML(taskset="mini-imagenet", num_tasks=10,
                          ways_num_classes_per_task=15,
                          shots_num_samples_per_class=5,
                          iterations=1500,
                          batch_size=32)
            # 2hr2 for 1500 epoch
        elif taskset == "fc100":
            tasksets_MAML(taskset="fc100", num_tasks=1000,
                          ways_num_classes_per_task=15,
                          shots_num_samples_per_class=1,
                          iterations=2000,
                          batch_size=32)
            # 57m for 5000 epoch

        # elif taskset == "cifarfs":
        #     tasksets_MAML(taskset="cifarfs", num_tasks=10,
        #                   ways_num_classes_per_task=15,
        #                   shots_num_samples_per_class=1,
        #                   iterations=5000,
        #                   batch_size=32)
        #     # 2 minutes for 200 epochs
        #     # 12 minutes for 1000 epochs
