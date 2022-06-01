import MAML_celeb, GBML_celeb, Meta_SGD_celeb
import MAML_benchmarking, GBML_benchmarking, Meta_SGD_benchmarking
import algorithm_benchmarking
import celebA_benchmarking
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv
import learn2learn as l2l
from CifarCNN import CifarCNN
from learn2learn.optim.transforms import MetaCurvatureTransform


def get_model(task, ways):
    if task == "omniglot":
        return "omniglot", l2l.vision.models.OmniglotFC(28 ** 2, ways)
    elif task == "omniglotCNN":
        return "omniglot", l2l.vision.models.OmniglotCNN(ways)
    elif task == "mini-imagenet":
        return "mini-imagenet", l2l.vision.models.MiniImagenetCNN(ways)
    elif task == "fc100":
        return "fc100", CifarCNN(output_size=ways)
    elif task == "celebA":
        return "celebA", l2l.vision.models.ResNet12(ways, hidden_size=2560)
    else:
        return "", l2l.vision.models.ResNet12(ways)


def get_algorithm(model, algorithm, fast_lr=0.5):
    if algorithm == "MAML":
        return l2l.algorithms.MAML(model, lr=fast_lr, allow_nograd=True)
    elif algorithm == "GBML":
        return l2l.algorithms.GBML(model, lr=fast_lr, transform=MetaCurvatureTransform, allow_nograd=True)
    elif algorithm == "MetaSGD":
        return l2l.algorithms.MetaSGD(model, lr=fast_lr)


# maybe try omniglotCNN for omniglot
def run_experiment(algorithm, taskset, tasks, ways, shots, iterations, batch_size, global_labels, save=True):
    save_file = f"./results/{algorithm}_{taskset}_{tasks}_{ways}_{shots}_{iterations}_{batch_size}_{global_labels}"

    taskset, model = get_model(taskset, ways)
    algorithm_ = get_algorithm(model, algorithm)

    if taskset == "celebA":
        data_plot, accuracy = celebA_benchmarking.main(model, algorithm_, tasks, ways, shots,
                                                       meta_batch_size=batch_size,
                                                       num_iterations=iterations, global_labels=global_labels)
    else:
        data_plot, accuracy = algorithm_benchmarking.main(model, algorithm_, taskset, tasks, ways, shots,
                                                          meta_batch_size=batch_size, num_iterations=iterations)
    # if algorithm == "MAML":
    #     if taskset == "celebA":
    #         data_plot, accuracy = MAML_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
    #                                               num_iterations=iterations, global_labels=global_labels)
    #     else:
    #         data_plot, accuracy = MAML_benchmarking.main(model, taskset, tasks, ways, shots, meta_batch_size=batch_size,
    #                                                      num_iterations=iterations)
    # elif algorithm == "GBML":
    #     if taskset == "celebA":
    #         data_plot, accuracy = GBML_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
    #                                               num_iterations=iterations, global_labels=global_labels)
    #     else:
    #         data_plot, accuracy = GBML_benchmarking.main(model, taskset, tasks, ways, shots, meta_batch_size=batch_size,
    #                                                      num_iterations=iterations)
    # elif algorithm == "MetaSGD":
    #     if taskset == "celebA":
    #         data_plot, accuracy = Meta_SGD_celeb.main(tasks, ways, shots, meta_batch_size=batch_size,
    #                                                   num_iterations=iterations, global_labels=global_labels)
    #     else:
    #         data_plot, accuracy = Meta_SGD_benchmarking.main(model, taskset, tasks, ways, shots,
    #                                                          meta_batch_size=batch_size, num_iterations=iterations)
    if save:
        with open(save_file, "w") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(data_plot)

        print(f"results of {algorithm} with {taskset} saved to {save_file}")


if __name__ == '__main__':
    algorithms = ["MAML", "GBML", "MetaSGD"]
    tasksets = ["celebA", "omniglot", "mini-imagenet", "fc100"]
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
    # tasksets = ["celebA", "omniglot", "mini-imagenet", "fc100"]
    # for algorithm in algorithms:
    #     for taskset in tasksets:
    #         print(f"running {algorithm} with {taskset}")
    #         if taskset == "celebA":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=50, shots=5, iterations=2,
    #                            batch_size=4, global_labels=False, save=False)
    #         else:
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=2,
    #                            batch_size=4, global_labels=True, save=False)

    # # minst? should be easy right
    # algorithms = ["MAML", "GBML", "MetaSGD"]
    # tasksets = ["omniglot", "mini-imagenet", "fc100", "celebA"]
    # for taskset in tasksets:
    #     for algorithm in algorithms:
    #         print(f"running {algorithm} with {taskset}")
    #         if taskset == "celebA":
    #             run_experiment(algorithm, taskset, tasks=500, ways=5000, shots=5, batch_size=128, iterations=200,
    #                            global_labels=False)
    #             pass
    #         elif taskset == "omniglot":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=200, ways=100, shots=5, iterations=200,
    #                            batch_size=32, global_labels=True)
    #             pass
    #         elif taskset == "mini-imagenet":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=750, ways=15, shots=5, iterations=200,
    #                            batch_size=32, global_labels=True)
    #             pass
    #         elif taskset == "fc100":
    #             run_experiment(algorithm=algorithm, taskset=taskset, tasks=2500, ways=20, shots=5, iterations=200,
    #                            batch_size=32, global_labels=True)
    #             pass

    # minst? should be easy right
    tasksets = ["mini-imagenet", "fc100"]
    algorithms = ["MAML", "GBML", "MetaSGD"]
    for taskset in tasksets:
        for algorithm in algorithms:
            print(f"running {algorithm} with {taskset}")
            if taskset == "celebA":
                run_experiment(algorithm, taskset, tasks=50, ways=250, shots=5, batch_size=32, iterations=500,
                               global_labels=False)
            else:
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=500,
                               batch_size=4, global_labels=True)
        # elif taskset == "omniglot":
        #     run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=500,
        #                    batch_size=4, global_labels=True)
        #     pass
        # elif taskset == "mini-imagenet":
        #     run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=500,
        #                    batch_size=4, global_labels=True)
        #     pass
        # elif taskset == "fc100":
        #     run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=500,
        #                    batch_size=4, global_labels=True)
        #     pass
        # else:
        #     run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, iterations=500,
        #                    batch_size=4, global_labels=True)
