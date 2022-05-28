import MAML_celeb, GBML_celeb, Meta_SGD_celeb
import MAML_benchmarking, GBML_benchmarking, Meta_SGD_benchmarking
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv


def run_experiment(algorithm, taskset, tasks, ways, shots, iterations, batch_size, global_labels):
    save_file = f"./results/{algorithm}/{taskset}/{tasks}_{ways}_{shots}_{iterations}_{batch_size}_{global_labels}"

    if algorithm == "MAML":
        if taskset == "celebA":
            data_plot, accuracy = MAML_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
                                                  num_iterations=iterations, global_labels=global_labels)
        else:
            data_plot, accuracy = MAML_benchmarking.main(taskset, tasks, ways, shots, meta_batch_size=batch_size,
                                                         num_iterations=iterations)
    elif algorithm == "GBML":
        if taskset == "celebA":
            data_plot, accuracy = GBML_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
                                                  num_iterations=iterations, global_labels=global_labels)
        else:
            data_plot, accuracy = GBML_benchmarking.main(taskset, tasks, ways, shots, meta_batch_size=batch_size,
                                                         num_iterations=iterations)
    elif algorithm == "Meta-SGD":
        if taskset == "celebA":
            data_plot, accuracy = Meta_SGD_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
                                                      num_iterations=iterations, global_labels=global_labels)
        else:
            data_plot, accuracy = Meta_SGD_benchmarking.main(taskset, tasks, ways, shots, meta_batch_size=batch_size,
                                                             num_iterations=iterations)
    with open(save_file, "w") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data_plot)

    print(f"results of {algorithm} with {taskset} saved to {save_file}")


if __name__ == '__main__':
    algorithms = ["MAML", "GBML", "Meta-SGD"]
    tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    for algorithm in algorithms:
        for taskset in tasksets:
            if not os.path.exists(f'./plots/{algorithm}/{taskset}'):
                os.makedirs(f'./plots/{algorithm}/{taskset}')
            if not os.path.exists(f'./results/{algorithm}/{taskset}'):
                os.makedirs(f'./results/{algorithm}/{taskset}')

    # # trial to see if it runs
    # algorithms = ["MAML", "GBML", "Meta-SGD"]
    # tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    # for algorithm in algorithms:
    #     for taskset in tasksets:
    #         if taskset == "celebA":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=5000, shots=5, iterations=5,
    #                            batch_size=16, global_labels=False)
    #
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=5000, shots=5, iterations=5,
    #                            batch_size=16, global_labels=True)
    #         elif taskset == "omniglot":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=100, shots=5, iterations=5,
    #                            batch_size=16, global_labels=True)
    #         elif taskset == "mini-imagenet":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=15, shots=5, iterations=5,
    #                            batch_size=16, global_labels=True)
    #         elif taskset == "fc100":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=1000, ways=15, shots=5, iterations=5,
    #                            batch_size=16, global_labels=True)

    algorithms = ["MAML", "GBML", "Meta-SGD"]
    tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    algorithms = ["GBML", "Meta-SGD"]
    tasksets = ["fc100"]
    for algorithm in algorithms:
        for taskset in tasksets:
            print(f"running {algorithm} with {taskset}")
            if taskset == "celebA":
                print(f"without global labels")
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=5000, shots=5, iterations=500,
                               batch_size=256, global_labels=False)
                print(f"with global labels")
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=5000, shots=5, iterations=500,
                               batch_size=256, global_labels=True)
            elif taskset == "omniglot":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=100, shots=5, iterations=500,
                               batch_size=128, global_labels=True)
            elif taskset == "mini-imagenet":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=15, shots=5, iterations=500,
                               batch_size=64, global_labels=True)
            elif taskset == "fc100":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=100, ways=20, shots=5, iterations=150,
                               batch_size=16, global_labels=True)
