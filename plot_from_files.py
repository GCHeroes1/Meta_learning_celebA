import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def collect_parameters(filename):
    # f"./results/celebA/{num_tasks}_{ways}_{shots}_"
    # f"{iterations}_{batch_size}_{str(global_labels)}",
    # / results / Meta - SGD / celebA / 10_500_5_30_256_False
    algorithm = filename.split("/")[-3]
    dataset = filename.split("/")[-2]
    hyperparameters = filename.split("/")[-1].split("_")
    if dataset == "celebA":
        num_tasks, ways, shots, iterations, batch_size, global_labels = hyperparameters
        print(global_labels)
        if global_labels == "False":
            global_labels = "out"
        else:
            global_labels = ""
    else:
        hyperparameters.append(None)
        num_tasks, ways, shots, iterations, batch_size, global_labels = hyperparameters
    return algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels


def calc_avg_std(data, rolling_average):
    train_err = [x[1] for x in data]
    train_acc = [c[2] for c in data]
    val_err = [v[3] for v in data]
    val_acc = [b[4] for b in data]
    test_err = [n[5] for n in data]
    test_acc = [m[6] for m in data]

    avg_train_err = pd.DataFrame(train_err).rolling(rolling_average).mean()[0]
    std_train_err = pd.DataFrame(train_err).rolling(rolling_average).std()[0] * .5

    avg_train_acc = pd.DataFrame(train_acc).rolling(rolling_average).mean()[0]
    std_train_acc = pd.DataFrame(train_acc).rolling(rolling_average).std()[0] * .5

    avg_val_err = pd.DataFrame(val_err).rolling(rolling_average).mean()[0]
    std_val_err = pd.DataFrame(val_err).rolling(rolling_average).std()[0] * .5

    avg_val_acc = pd.DataFrame(val_acc).rolling(rolling_average).mean()[0]
    std_val_acc = pd.DataFrame(val_acc).rolling(rolling_average).std()[0] * .5

    avg_test_err = pd.DataFrame(test_err).rolling(rolling_average).mean()[0]
    std_test_err = pd.DataFrame(test_err).rolling(rolling_average).std()[0] * .5

    avg_test_acc = pd.DataFrame(test_acc).rolling(rolling_average).mean()[0]
    std_test_acc = pd.DataFrame(test_acc).rolling(rolling_average).std()[0] * .5

    return avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, \
           std_val_acc, avg_test_err, std_test_err, avg_test_acc, std_test_acc


def plotting_averages(filename, data_plot, rolling_average=50):
    algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels = collect_parameters(filename)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    N = np.arange(len(data_plot))
    avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, std_val_acc, \
    avg_test_err, std_test_err, avg_test_acc, std_test_acc = calc_avg_std(data_plot, rolling_average)

    plt.style.use('ggplot')
    title = (f"{algorithm}, {dataset} dataset, train and val accuracy & error for {num_tasks} tasks with {ways} classes"
             f"\nbatch size is {batch_size}, over {iterations} iterations")
    if dataset == "celebA":
        title += f", with{global_labels} global labels"
    plt.suptitle(title, fontsize=14)

    ax[0].plot(N, avg_train_acc, alpha=0.5, color='blue', label='Training Accuracy', linewidth=1.0)
    ax[0].plot(N, avg_val_acc, alpha=0.5, color='red', label='Validation Accuracy', linewidth=1.0)
    ax[0].plot(N, avg_test_acc, alpha=0.5, color='green', label='Testing Accuracy', linewidth=1.0)
    ax[0].fill_between(N, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='blue', alpha=0.3)
    ax[0].fill_between(N, avg_val_acc - std_val_acc, avg_val_acc + std_val_acc, color='red', alpha=0.3)
    ax[0].fill_between(N, avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, color='green', alpha=0.3)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Iterations")
    ax[0].legend(loc='best')

    ax[1].plot(N, avg_train_err, alpha=0.5, color='blue', label='Training Error', linewidth=1.0)
    ax[1].plot(N, avg_val_err, alpha=0.5, color='red', label='Validation Error', linewidth=1.0)
    ax[1].plot(N, avg_test_err, alpha=0.5, color='green', label='Testing Error', linewidth=1.0)
    ax[1].fill_between(N, avg_train_err - std_train_err, avg_train_err + std_train_err, color='blue', alpha=0.3)
    ax[1].fill_between(N, avg_val_err - std_val_err, avg_val_err + std_val_err, color='red', alpha=0.3)
    ax[1].fill_between(N, avg_test_err - std_test_err, avg_test_err + std_test_err, color='green', alpha=0.3)
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Iterations")
    ax[1].legend(loc='best')
    # plt.tight_layout(pad=1.5)
    # plt.savefig(f'./Figs/average_actions_rewards_10000.png')

    save_file = f'./plots/{algorithm}/{dataset}/{str(num_tasks)}_{str(ways)}_{shots}_{batch_size}_{iterations}'
    if dataset == "celebA":
        if global_labels == "out":
            save_file += "_False"
        else:
            save_file += "_True"
    save_file += "_v2.png"
    print(save_file)
    plt.savefig(save_file)
    plt.show()
    plt.clf()
    return


if __name__ == '__main__':
    # file = "./results/fc100/10_15_1_5000_32"
    # file = "./results/fc100/1000_15_1_2000_32"
    # file = "./results/omniglot/10_100_1_1500_32"
    # file = "./results/mini-imagenet/10_15_5_1500_32"
    # file = "./results/celebA/10_500_20_1000_128_False"
    # file = "./results/celebA/10_500_20_200_128_True"
    # file = "./results/Meta-SGD/celebA/10_500_5_30_256_False"
    # file = "./results/GBML/celebA/10_500_5_30_256_False"
    file = "./results/MAML/celebA/10_500_5_100_256_False"
    data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if len(row) > 0:
                data.append(list(map(float, row)))
                line_count += 1
        # print(f"processed {line_count} lines")
    data = data[:-1]
    plotting_averages(filename=file, data_plot=data, rolling_average=int(len(data) / 20))
