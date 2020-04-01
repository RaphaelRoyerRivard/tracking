from main import read_ground_truth, visualize_bounding_boxes


def visualize_result(result_file_path):
    f = open(result_file_path)
    lines = f.readlines()
    f.close()
    predictions = []
    for line in lines:
        predictions.append(tuple(map(int, map(float, line[1:-2].split(", ")))))
    gt_folder = f"data/{result_file_path.split('/')[-1].split('_')[0]}"
    gt_file_path = gt_folder + "/groundtruth.txt"
    gt = read_ground_truth(gt_file_path)
    visualize_bounding_boxes(gt_folder, gt, predictions, delay=50)


if __name__ == '__main__':
    visualize_result("results/hand_CSRT.txt")
