import algorithm_benchmarking
import celebA_benchmarking
import os
import csv
import learn2learn as l2l
from CifarCNN import CifarCNN
from learn2learn.optim.transforms import MetaCurvatureTransform, KroneckerTransform


def get_model(task, ways):
    if task == "omniglot":
        return task, l2l.vision.models.OmniglotFC(28 ** 2, ways)
    elif task == "omniglotCNN":
        return "omniglot", l2l.vision.models.OmniglotCNN(ways)
    elif task == "mini-imagenet" or task == "tiered-imagenet":
        return task, l2l.vision.models.ResNet12(ways)
    elif task == "fc100":
        return task, CifarCNN(output_size=ways)
    elif task == "celebA":
        return task, l2l.vision.models.ResNet12(ways, hidden_size=5760)
    else:
        return task, l2l.vision.models.ResNet12(ways)


def get_algorithm(model, algorithm, fast_lr=0.5):
    if algorithm == "MAML":
        return l2l.algorithms.MAML(model, lr=fast_lr, first_order=False, allow_nograd=True)
    elif algorithm == "MetaCurvature":
        return l2l.algorithms.GBML(model, lr=fast_lr, transform=MetaCurvatureTransform, allow_nograd=True)
    elif algorithm == "MetaSGD":
        return l2l.algorithms.MetaSGD(model, lr=fast_lr)
    elif algorithm == "MetaKFO":
        kronecker_transform = KroneckerTransform(l2l.nn.KroneckerLinear)
        return l2l.algorithms.GBML(model, lr=fast_lr, transform=kronecker_transform, allow_nograd=True)


def run_experiment(algorithm, taskset, tasks, ways, shots, adaptation_steps, iterations, batch_size, global_labels,
                   save=True):
    save_file = f"{RESULTS_DIR}/{algorithm}_{taskset}_{tasks}_{ways}_{shots}_{adaptation_steps}_{iterations}_{batch_size}_{global_labels}"
    if os.path.exists(save_file):
        return

    taskset, model = get_model(taskset, ways)
    algorithm_ = get_algorithm(model, algorithm)

    if taskset == "celebA":
        data_plot = celebA_benchmarking.main(model, algorithm_, tasks, ways, shots, adaptation_steps,
                                             meta_batch_size=batch_size, num_iterations=iterations,
                                             global_labels=global_labels)
    else:
        data_plot = algorithm_benchmarking.main(model, algorithm_, taskset, tasks, ways, shots, adaptation_steps,
                                                meta_batch_size=batch_size, num_iterations=iterations)

    if save:
        with open(save_file, "w") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(data_plot)

        print(f"results of {algorithm} with {taskset} saved to {save_file}")


if __name__ == '__main__':
    RESULTS_DIR = './results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # trial to see if it runs
    algorithms = ["MAML", "MetaKFO", "MetaSGD", "MetaCurvature"]
    tasksets = ["celebA", "omniglot", "mini-imagenet", "fc100", "tiered-imagenet"]
    for algorithm in algorithms:
        for taskset in tasksets:
            print(f"running {algorithm} with {taskset}")
            if taskset == "celebA":
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=10, ways=50, shots=5, adaptation_steps=1,
                               iterations=2, batch_size=1, global_labels=False, save=False)
            else:
                run_experiment(algorithm=algorithm, taskset=taskset, tasks=50, ways=5, shots=5, adaptation_steps=1,
                               iterations=2, batch_size=1, global_labels=True, save=False)

    # main test
    algorithms = ["MAML", "MetaKFO", "MetaSGD", "MetaCurvature"]
    tasksets = ["celebA", "omniglot", "mini-imagenet", "fc100", "tiered-imagenet"]
    for taskset in tasksets:
        for algorithm in algorithms:
            print(f"running {algorithm} with {taskset}")
            try:
                if taskset == "celebA":
                    run_experiment(algorithm, taskset, tasks=1000, ways=5000, shots=5, adaptation_steps=1,
                                   batch_size=32, iterations=1000, global_labels=False)
                else:
                    run_experiment(algorithm, taskset, tasks=1000, ways=5, shots=5, adaptation_steps=1, batch_size=32,
                                   iterations=500, global_labels=True)
            except:
                print("failed")

    # extra test
    algorithms = ["MAML", "MetaKFO", "MetaSGD", "MetaCurvature"]
    tasksets = ["celebA"]
    for taskset in tasksets:
        for algorithm in algorithms:
            print(f"running {algorithm} with {taskset}")
            try:
                if taskset == "celebA":
                    run_experiment(algorithm, taskset, tasks=1000, ways=5000, shots=5, adaptation_steps=1,
                                   batch_size=32, iterations=2000, global_labels=False)
                else:
                    run_experiment(algorithm, taskset, tasks=1000, ways=5, shots=5, adaptation_steps=1, batch_size=32,
                                   iterations=500, global_labels=True)
            except:
                print("failed")
