import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import glob
import time


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def collect_parameters(filename):
    # f"./results/celebA/{num_tasks}_{ways}_{shots}_"
    # f"{iterations}_{batch_size}_{str(global_labels)}",
    # / results / Meta - SGD / celebA / 10_500_5_30_256_False
    # ./plots/Meta-SGD_omniglot_50_10_5_64_2_v2.png
    # algorithm = filename.split("\\")[-3]
    # dataset = filename.split("\\")[-2]
    hyperparameters = filename.split("\\")[-1].split("_")
    algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels = hyperparameters
    # if dataset == "celebA":
    #     if global_labels == "False":
    #         global_labels = "out"
    #     else:
    #         global_labels = ""
    return algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels


def calc_avg_std(data, rolling_average):
    train_err = [x[1] for x in data]
    train_acc = [c[2] for c in data]
    val_err = [v[3] for v in data]
    val_acc = [b[4] for b in data]
    # test_err = [n[5] for n in data]
    # test_acc = [m[6] for m in data]

    avg_train_err = pd.DataFrame(train_err).rolling(rolling_average).mean()[0]
    std_train_err = pd.DataFrame(train_err).rolling(rolling_average).std()[0] * .5

    avg_train_acc = pd.DataFrame(train_acc).rolling(rolling_average).mean()[0]
    std_train_acc = pd.DataFrame(train_acc).rolling(rolling_average).std()[0] * .5

    avg_val_err = pd.DataFrame(val_err).rolling(rolling_average).mean()[0]
    std_val_err = pd.DataFrame(val_err).rolling(rolling_average).std()[0] * .5

    avg_val_acc = pd.DataFrame(val_acc).rolling(rolling_average).mean()[0]
    std_val_acc = pd.DataFrame(val_acc).rolling(rolling_average).std()[0] * .5

    # avg_test_err = pd.DataFrame(test_err).rolling(rolling_average).mean()[0]
    # std_test_err = pd.DataFrame(test_err).rolling(rolling_average).std()[0] * .5
    #
    # avg_test_acc = pd.DataFrame(test_acc).rolling(rolling_average).mean()[0]
    # std_test_acc = pd.DataFrame(test_acc).rolling(rolling_average).std()[0] * .5

    return avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, \
           std_val_acc


def resave_file(filename, data):
    algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels = collect_parameters(filename)
    save_file = f"./results/{algorithm}_{dataset}_{num_tasks}_{ways}_{shots}_{iterations}_{batch_size}_{global_labels}"
    if os.path.exists(save_file):
        return
    else:
        with open(save_file, "w") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(data)

        print(f"results of {algorithm} with {dataset} saved to {save_file}")


def plotting_averages(filename, data_plot, rolling_average=50):
    algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels = collect_parameters(filename)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    N = np.arange(len(data_plot))
    avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, std_val_acc, \
        = calc_avg_std(data_plot, rolling_average)

    save_file = f'./plots/{algorithm}_{dataset}_{str(num_tasks)}_{str(ways)}_{shots}_{batch_size}_{iterations}_{global_labels}'
    # if dataset == "celebA":
    #     if global_labels == "out":
    #         save_file += "_False"
    #     else:
    #         save_file += "_True"
    # save_file += "_v2.png"
    if os.path.exists(save_file):
        return

    plt.style.use('ggplot')
    time.sleep(0.1)
    title = (
        f"{algorithm}, {dataset} dataset, train and val accuracy & error for {str(int(num_tasks))} tasks with {ways} classes"
        f"\nand {shots} shots, meta batch size is {batch_size}, trained for {iterations} iterations")
    if not global_labels:
        title += f", without global labels"
    else:
        title += f", with global labels"
    plt.suptitle(title, fontsize=14)

    ax[0].plot(N, avg_train_acc, alpha=0.5, color='blue', label='Training Accuracy', linewidth=1.0)
    ax[0].plot(N, avg_val_acc, alpha=0.5, color='red', label='Validation Accuracy', linewidth=1.0)
    # ax[0].plot(N, avg_test_acc, alpha=0.5, color='green', label='Testing Accuracy', linewidth=1.0)
    ax[0].fill_between(N, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='blue', alpha=0.3)
    ax[0].fill_between(N, avg_val_acc - std_val_acc, avg_val_acc + std_val_acc, color='red', alpha=0.3)
    # ax[0].fill_between(N, avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, color='green', alpha=0.3)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Iterations")
    # ax[0].set_ylim([0, 0.5])
    ax[0].legend(loc='best')

    ax[1].plot(N, avg_train_err, alpha=0.5, color='blue', label='Training Error', linewidth=1.0)
    ax[1].plot(N, avg_val_err, alpha=0.5, color='red', label='Validation Error', linewidth=1.0)
    # ax[1].plot(N, avg_test_err, alpha=0.5, color='green', label='Testing Error', linewidth=1.0)
    ax[1].fill_between(N, avg_train_err - std_train_err, avg_train_err + std_train_err, color='blue', alpha=0.3)
    ax[1].fill_between(N, avg_val_err - std_val_err, avg_val_err + std_val_err, color='red', alpha=0.3)
    # ax[1].fill_between(N, avg_test_err - std_test_err, avg_test_err + std_test_err, color='green', alpha=0.3)
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Iterations")
    ax[1].legend(loc='best')

    print("done", save_file)
    plt.savefig(save_file)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close("all")
    return


if __name__ == '__main__':
    import sys

    data_path = "./results/"
    for result_file in glob.glob(data_path + os.path.sep + "*"):
        data = []
        with open(result_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if len(row) > 0:
                    data.append(list(map(float, row)))
                    line_count += 1
            # print(f"processed {line_count} lines")
        try:
            plotting_averages(filename=result_file, data_plot=data, rolling_average=int(5))
        except:
            print(result_file, "failed")

    # for algorithm_dir in glob.glob(data_path + os.path.sep + "*"):
    #     for dataset_dir in glob.glob(algorithm_dir + os.path.sep + "*"):
    #         for result_file in glob.glob(dataset_dir + os.path.sep + "*"):
    #             data = []
    #             with open(result_file) as csv_file:
    #                 csv_reader = csv.reader(csv_file, delimiter=",")
    #                 line_count = 0
    #                 for row in csv_reader:
    #                     if len(row) > 0:
    #                         data.append(list(map(float, row)))
    #                         line_count += 1
    #                 # print(f"processed {line_count} lines")
    #             # data = data[:-1]
    #             try:
    #                 resave_file(result_file, data)
    #             except:
    #                 print(result_file, "failed")
