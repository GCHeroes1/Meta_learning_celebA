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
    return algorithm, dataset, num_tasks, ways, shots, iterations, batch_size, global_labels


def collect_new_parameters(filename):
    hyperparameters = filename.split("\\")[-1].split("_")
    algorithm, dataset, num_tasks, ways, shots, adaptation_steps, iterations, batch_size, global_labels = hyperparameters
    return algorithm, dataset, num_tasks, ways, shots, adaptation_steps, iterations, batch_size, global_labels


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
    algorithm, dataset, num_tasks, ways, shots, adaptation_steps, iterations, batch_size, global_labels = \
        collect_new_parameters(filename)

    N = np.arange(len(data_plot))
    avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, std_val_acc, \
    avg_test_err, std_test_err, avg_test_acc, std_test_acc = calc_avg_std(data_plot, rolling_average)

    save_file = f'{PLOTS_DIR}/{algorithm}_{dataset}_{str(num_tasks)}_{str(ways)}_{shots}_{adaptation_steps}_' \
                f'{batch_size}_{iterations}_{global_labels}.png'

    # if os.path.exists(save_file):
    #     return

    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    time.sleep(0.1)
    title = (
        f"{algorithm}, {dataset} dataset, train and val accuracy & error for {str(int(num_tasks))} tasks {ways} classes"
        f"\nand {shots} shots, {adaptation_steps} adaptation step, meta batch size is {batch_size}, trained for {iterations} iterations")
    if global_labels == "False" or global_labels == "False2":
        title += f", without global labels"
    else:
        title += f", with global labels"
    plt.suptitle(title, fontsize=14)

    ax[0].plot(N, avg_train_acc, alpha=0.5, color='blue', label='Training Accuracy', linewidth=1.0)
    ax[0].plot(N, avg_val_acc, alpha=0.5, color='red', label='Validation Accuracy', linewidth=1.0)
    ax[0].plot(N, avg_test_acc, alpha=0.5, color='green', label='Testing Accuracy', linewidth=1.0)
    ax[0].fill_between(N, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='blue', alpha=0.3)
    ax[0].fill_between(N, avg_val_acc - std_val_acc, avg_val_acc + std_val_acc, color='red', alpha=0.3)
    ax[0].fill_between(N, avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, color='green', alpha=0.3)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Iterations")
    ax[0].set_yticks(np.arange(0, 1, 0.1))
    ax[0].set_yticks(ax[0].get_yticks()[::1])
    # ax[0].set_ylim([0, 0.6])
    ax[0].legend(loc='best')

    ax[1].plot(N, avg_train_err, alpha=0.5, color='blue', label='Training Error', linewidth=1.0)
    ax[1].plot(N, avg_val_err, alpha=0.5, color='red', label='Validation Error', linewidth=1.0)
    ax[1].plot(N, avg_test_err, alpha=0.5, color='green', label='Testing Error', linewidth=1.0)
    ax[1].fill_between(N, avg_train_err - std_train_err, avg_train_err + std_train_err, color='blue', alpha=0.3)
    ax[1].fill_between(N, avg_val_err - std_val_err, avg_val_err + std_val_err, color='red', alpha=0.3)
    ax[1].fill_between(N, avg_test_err - std_test_err, avg_test_err + std_test_err, color='green', alpha=0.3)
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Iterations")
    # ax[1].set_yscale("log")
    ax[1].legend(loc='best')

    print("done", save_file)
    # plt.savefig(save_file)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close("all")
    return


def plotting_averages_single(filename, data_plot, rolling_average=5):
    algorithm, dataset, num_tasks, ways, shots, adaptation_steps, iterations, batch_size, global_labels = \
        collect_new_parameters(filename)

    N = np.arange(len(data_plot))
    avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, std_val_acc, \
    avg_test_err, std_test_err, avg_test_acc, std_test_acc = calc_avg_std(data_plot, rolling_average)

    save_file = f'{PLOTS_DIR}/{algorithm}_{dataset}_{str(num_tasks)}_{str(ways)}_{shots}_{adaptation_steps}_' \
                f'{batch_size}_{iterations}_{global_labels}.png'

    if os.path.exists(save_file):
        return

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(6, 5))
    time.sleep(0.1)
    title = (f"{algorithm} on {dataset}")
    if global_labels == "False":
        title += f" without global labels \n"
    else:
        title += f" with global labels \n"
    title += f"Training and validation accuracy for {str(int(num_tasks))} tasks, {ways}-way {shots}-shot"
    ax.set_title(title, loc='center', wrap=True)

    ax.plot(N, avg_train_acc, alpha=1, color='blue', label='Training Accuracy', linewidth=1.0)
    ax.plot(N, avg_val_acc, alpha=1, color='red', label='Validation Accuracy', linewidth=1.0)
    ax.plot(N, avg_test_acc, alpha=1, color='green', label='Testing Accuracy', linewidth=1.0)
    ax.fill_between(N, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='blue', alpha=0.2)
    ax.fill_between(N, avg_val_acc - std_val_acc, avg_val_acc + std_val_acc, color='red', alpha=0.2)
    ax.fill_between(N, avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, color='green', alpha=0.2)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Iterations")
    ax.set_ylim(bottom=0.)
    ax.set_yticks(ax.get_yticks()[::1])
    # ax[0].set_ylim([0, 0.6])
    plt.tight_layout(rect=[-0.01, -0.02, 1, 0.95])
    ax.legend(loc='best')

    print("done", save_file)
    plt.savefig(save_file)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close("all")
    return


def plotting_algorithm_grid(dataset="celebA", iterations=1000, rolling_average=25):
    global_labels = True
    if dataset == "celebA":
        global_labels = False
    save_file = f'{PLOTS_DIR}/{dataset}_{str(1000)}_{str(5)}_{5}_{1}_{32}_{iterations}_{global_labels}.png'
    # if os.path.exists(save_file):
    #     return

    algorithms = ["MAML", "MetaKFO", "MetaSGD", "MetaCurvature"]
    big_data = []
    for algorithm in algorithms:
        for result_file in glob.glob(RESULTS_DIR + os.path.sep + "*"):
            algorithm_, dataset_, num_tasks_, ways_, shots_, adaptation_steps_, iterations_, batch_size_, global_labels_ = \
                collect_new_parameters(result_file)
            if algorithm_ == algorithm and dataset_ == dataset and int(iterations_) == iterations:
                data = []
                with open(result_file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    line_count = 0
                    for row in csv_reader:
                        if len(row) > 0:
                            data.append(list(map(float, row)))
                            line_count += 1
                big_data.append((algorithm, data))
    if len(big_data) != 4:
        return
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    title = f"{dataset}"
    if global_labels == False:
        title += f" without global labels \n"
    else:
        title += f" with global labels \n"
    title += f"Training and validation accuracy for {str(1000)}-task {5}-way {5}-shot"
    plt.suptitle(title, fontsize=20, wrap=True)
    for i, ax in enumerate(axs.reshape(-1)):
        N = np.arange(len(big_data[i][1]))
        avg_train_err, std_train_err, avg_train_acc, std_train_acc, avg_val_err, std_val_err, avg_val_acc, std_val_acc, \
        avg_test_err, std_test_err, avg_test_acc, std_test_acc = calc_avg_std(big_data[i][1], rolling_average)

        ax.plot(N, avg_train_acc, alpha=1, color='blue', label='Training Accuracy', linewidth=1.0)
        ax.plot(N, avg_val_acc, alpha=1, color='red', label='Validation Accuracy', linewidth=1.0)
        ax.plot(N, avg_test_acc, alpha=1, color='green', label='Testing Accuracy', linewidth=1.0)
        ax.fill_between(N, avg_train_acc - std_train_acc, avg_train_acc + std_train_acc, color='blue', alpha=0.2)
        ax.fill_between(N, avg_val_acc - std_val_acc, avg_val_acc + std_val_acc, color='red', alpha=0.2)
        ax.fill_between(N, avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, color='green', alpha=0.2)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Iterations")
        ax.set_ylim(bottom=0.)
        ax.set_yticks(ax.get_yticks()[::1])
        ax.set_title(big_data[i][0])
        # ax.tight_layout(rect=[-0.01, -0.02, 1, 0.95])
        # ax[0].set_ylim([0, 0.6])
        ax.legend(loc='best')
    # plt.tight_layout(rect=[-0.01, -0.02, 1, 0.95])
    plt.tight_layout()
    print("done", save_file)
    plt.savefig(save_file)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close("all")


if __name__ == '__main__':
    RESULTS_DIR = './results_final'
    PLOTS_DIR = './plots_actual_final'

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # data_path = RESULTS_DIR
    # for result_file in glob.glob(data_path + os.path.sep + "*"):
    #     data = []
    #     with open(result_file) as csv_file:
    #         csv_reader = csv.reader(csv_file, delimiter=",")
    #         line_count = 0
    #         for row in csv_reader:
    #             if len(row) > 0:
    #                 data.append(list(map(float, row)))
    #                 line_count += 1
    #         # print(f"processed {line_count} lines")
    #     plotting_averages_single(filename=result_file, data_plot=data, rolling_average=int(50))
    #     # break
    #     # try:
    #     #     plotting_averages(filename=result_file, data_plot=data, rolling_average=int(5))
    #     # except:
    #     #     print(result_file, "failed")

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

    # plotting_algorithm_grid()
    datasets = ["celebA"]
    for dataset in datasets:
        plotting_algorithm_grid(dataset=dataset, iterations=500, rolling_average=40)
