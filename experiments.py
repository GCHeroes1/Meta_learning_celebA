import MAML_celeb, GBML_celeb, Meta_SGD_celeb
import MAML_benchmarking, GBML_benchmarking, Meta_SGD_benchmarking
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv


# maybe try omniglotCNN for omniglot
def run_experiment(algorithm, taskset, tasks, ways, shots, iterations, batch_size, global_labels, save=True):
    save_file = f"./results/{algorithm}_{taskset}_{tasks}_{ways}_{shots}_{iterations}_{batch_size}_{global_labels}"

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
    elif algorithm == "MetaSGD":
        if taskset == "celebA":
            data_plot, accuracy = Meta_SGD_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
                                                      num_iterations=iterations, global_labels=global_labels)
        else:
            data_plot, accuracy = Meta_SGD_benchmarking.main(taskset, tasks, ways, shots, meta_batch_size=batch_size,
                                                             num_iterations=iterations)
    if save:
        with open(save_file, "w") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(data_plot)

        print(f"results of {algorithm} with {taskset} saved to {save_file}")


if __name__ == '__main__':
    algorithms = ["MAML", "GBML", "MetaSGD"]
    tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    if not os.path.exists(f'./plots'):
        os.makedirs(f'./plots')
    if not os.path.exists(f'./results'):
        os.makedirs(f'./results')
    # for algorithm in algorithms:
    #     for taskset in tasksets:
    #         if not os.path.exists(f'./plots/{algorithm}/{taskset}'):
    #             os.makedirs(f'./plots/{algorithm}/{taskset}')
    #         if not os.path.exists(f'./results/{algorithm}/{taskset}'):
    #             os.makedirs(f'./results/{algorithm}/{taskset}')

    # # trial to see if it runs
    # algorithms = ["MAML", "GBML", "MetaSGD"]
    # tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    # for algorithm in algorithms:
    #     for taskset in tasksets:
    #         print(f"running {algorithm} with {taskset}")
    #         if taskset == "celebA":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=50, shots=5, iterations=2,
    #                            batch_size=512, global_labels=True, save=False)
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=500, ways=5000, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=False, save=False)
    #             #
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=50, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=True, save=False)
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=500, ways=5000, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=True, save=False)
    #         elif taskset == "omniglot":
    #             pass
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=10, shots=5, iterations=2,
    #                            batch_size=4, global_labels=True, save=True)
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=25, ways=10, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=True, save=False)
    #         elif taskset == "mini-imagenet":
    #             pass
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=750, ways=15, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=True, save=False)
    #         elif taskset == "fc100":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=500, ways=20, shots=20, iterations=2,
    #                            batch_size=2, global_labels=True, save=True)
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=500, ways=20, shots=50, iterations=2,
    #                            batch_size=2, global_labels=True, save=True)
    #
    #             # run_experiment(algorithm=algorithm, taskset=taskset, tasks=250, ways=10, shots=5, iterations=2,
    #             #                batch_size=4, global_labels=True, save=False)

    algorithms = ["MAML", "GBML", "MetaSGD"]
    tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    for taskset in tasksets:
        for algorithm in algorithms:
            print(f"running {algorithm} with {taskset}")
            if taskset == "celebA":
                run_experiment(algorithm, taskset, tasks=500, ways=5000, shots=5, batch_size=128, iterations=500,
                               global_labels=False)
                pass
            elif taskset == "omniglot":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=200, ways=100, shots=5, iterations=500,
                               batch_size=32, global_labels=True)
                pass
            elif taskset == "mini-imagenet":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=750, ways=15, shots=5, iterations=70,
                               batch_size=32, global_labels=True)
                pass
            elif taskset == "fc100":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=2500, ways=20, shots=5, iterations=500,
                               batch_size=32, global_labels=True)
                pass
