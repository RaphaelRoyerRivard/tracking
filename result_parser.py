import os
import numpy as np
from matplotlib import pyplot as plt


def autolabel(rects, ax, precision, size):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(('{:.' + str(precision) + 'f}').format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=size)


def parse_results():
    frames = {}
    results = {}
    for path, subfolders, files in os.walk("data"):
        if "groundtruth.txt" not in files:
            continue

        f = open(f'{path}/groundtruth.txt', 'r')
        lines = f.readlines()
        f.close()

        dataset = path.split("\\")[-1]
        frames[dataset] = len(lines) - 1
        results[dataset] = {}

    for threshold in [30, 50, 70]:
        f = open(f'results_{threshold}/result.txt', 'r')
        lines = f.readlines()
        f.close()

        trackers = []
        for line in lines:
            values = line.split(";")
            dataset = values[0]
            tracker = values[1]
            if tracker not in trackers:
                trackers.append(tracker)
            accuracy = float(values[2])
            robustness = float(values[3])
            duration = float(values[4])
            fps = frames[dataset] / duration
            if tracker not in results[dataset].keys():
                results[dataset][tracker] = {}
            results[dataset][tracker][threshold] = (accuracy, robustness, fps)

    values = []
    for i, dataset in enumerate(results.keys()):
        values.append([])
        for j, tracker in enumerate(results[dataset].keys()):
            values[i].append([])
            for threshold in results[dataset][tracker].keys():
                v = results[dataset][tracker][threshold]
                v = [v[0], v[1], v[2]]
                values[i][j].append(v)
    values = np.array(values)   # (16, 7, 3, 3) => datasets, methods, IoU thresholds, metrics
    bar_width = 0.11
    value_types = ["Accuracy", "Robustness", "FPS"]
    for t in range(len(value_types)):

        # Graph 1: separated for image pair
        fig, ax = plt.subplots()
        plt.grid(True, axis='y', zorder=0)
        x = np.arange(values.shape[0])
        for j in range(values.shape[1]):
            bar_values = values[:, j, 1, t]  # Using only the 50% IoU threshold
            label = trackers[j]
            ax.bar(x + bar_width * j - bar_width * 3, bar_values, bar_width, label=label, zorder=2)
        ax.set_xlabel("Video")
        ax.set_ylabel("Frames per second" if t == 2 else "Percentage")
        ax.set_title(value_types[t] + " for IoU threshold of 50%")
        ax.set_xticks(x)
        ax.set_xticklabels(list(results.keys()))
        ax.legend()
        fig.tight_layout()
        plt.show()

        # Graph 2: separated for image pair
        fig, ax = plt.subplots()
        plt.grid(True, axis='y', zorder=0)
        x = np.arange(values.shape[1])
        rects = []
        for j in x:
            bar_values = [values[:, j, threshold, t].mean() for threshold in range(values.shape[-2])]
            bars_x = [j + (threshold - 1) * bar_width * 1.5 for threshold in range(values.shape[-2])]
            rects.append(ax.bar(bars_x, bar_values, bar_width, zorder=2))
        ax.set_xlabel("Tracking method")
        ax.set_ylabel("Frames per second" if t == 2 else "Percentage")
        ax.set_title(value_types[t] + " for IoU thresholds of 30%, 50% and 70%")
        ax.set_xticks(x)
        ax.set_xticklabels(trackers)
        for rect in rects:
            autolabel(rect, ax, precision=2, size=10)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parse_results()
