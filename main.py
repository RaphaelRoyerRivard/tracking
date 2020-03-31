import os
import time
import cv2


def bb_iou(box1, box2):
    """
    computes intersection over union (iou) between 2 bounding boxes.
    :param box1: prediction box.
    :param box2: gt box.
    :return: iou.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    # object is lost, no box
    if x_right == 0 or y_bottom == 0:
        return 0.0
    overlap_area = (x_right - x_left) * (y_bottom - y_top)  # intersection
    bb1_area = box1[2] * box1[3]
    bb2_area = box2[2] * box2[3]
    combined_area = bb1_area + bb2_area - overlap_area  # union
    iou = overlap_area / float(combined_area)
    return iou


def evaluate(predictions, ground_truth, iou_cutoff=0.5):
    """
    evaluation method.
    :param predictions: dict of the predictions as tuples.
    :param ground_truth: dict of the gt as tuples.
    :param iou_cutoff: value at which an iou is considered as true positives.
    :return: accuracy and robustness metrics.
    accuracy = ratio of the number of times the object was correctly tracked across all frames.
    robustness = precision of the tracking when the object was correctly tracked.
    """
    assert len(predictions) == len(ground_truth)
    tp = 0
    mean_iou = 0.0
    for i in range(len(ground_truth)):
        prediction = predictions[i]
        gt = ground_truth[i]
        iou = bb_iou(prediction, gt)

        if iou >= iou_cutoff:
            tp += 1
            mean_iou += iou
    return float(tp) / len(ground_truth), float(mean_iou) / float(tp)


def get_frames(folder):
    """
    get the full path to all the frames in the video sequence.
    :param folder: path to the folder containing the frames of the video sequence.
    :return: list of the name of all the frames.
    """
    names = os.listdir(folder)
    frames = [os.path.join(folder, n) for n in names if n.endswith('.jpg')]
    frames.sort()
    return frames


def init_tracker(gt):
    """
    get the object location in the first frame.
    :param gt: box location for each frame (output of read_ground_truth).
    :return: location of the object in the first frame.
    """
    return gt[0]


def read_ground_truth(path):
    """
    reads ground-truth and returns it as a numpy array.
    :param path: path to groundtruth.txt file.
    :return: dict of the 4 the coordinates for top-left corner and width/height as tuple.
    """
    ground_truth = {}
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            x, y, width, height = line.split(',')
            x = int(float(x))
            y = int(float(y))
            width = int(float(width))
            height = int(float(height))
            ground_truth[i] = (x, y, width, height)
    return ground_truth


def test_ground_truth(folder, gt):
    """
    use this function to see the ground-truth boxes on the sequence.
    :param folder: path to the folder containing the frames of the video sequence.
    :param gt: box location for each frame (output of read_ground_truth).
    :return: void
    """
    frames = get_frames(folder)

    for i, frame in enumerate(frames):
        box = gt[i]
        frame = cv2.imread(frame, cv2.IMREAD_COLOR)
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 0, 255))
        cv2.imshow('sequence', frame)
        cv2.waitKey(delay=24)


def track(folder, first_bb, tracker):
    """
    code for your tracker.
    :param folder: path to the folder containing the frames of the video sequence.
    :param first_bb: box location for the first frame (output of init_tracker).
    :param tracker: the cv2 tracker object.
    :return: list with an entry for each frame being a tuple (x, y, width, height)
    """
    frames = get_frames(folder)
    predictions = [first_bb]

    for i, frame in enumerate(frames):
        frame = cv2.imread(frame, cv2.IMREAD_COLOR)
        if i == 0:
            tracker.init(frame, first_bb)
            continue
        (success, bb) = tracker.update(frame)
        # print(success, bb)
        predictions.append(bb)

    return predictions


def main():
    output_folder = 'results'
    data_folder = "data"
    for path, subfolders, files in os.walk(data_folder):
        print(path)
        if "groundtruth.txt" not in files:
            continue
        dataset = path.split("\\")[-1]
        path_gt = f'{path}/groundtruth.txt'
        trackers = {
            "CSRT": cv2.TrackerCSRT_create(),
            "KCF": cv2.TrackerKCF_create(),
            "Boosting": cv2.TrackerBoosting_create(),
            "MIL": cv2.TrackerMIL_create(),
            "TLD": cv2.TrackerTLD_create(),
            "MedianFlow": cv2.TrackerMedianFlow_create(),
            "MOSSE": cv2.TrackerMOSSE_create(),
        }
        gt = read_ground_truth(path_gt)
        # test_ground_truth(frames_folder, gt)
        for tracker_name, tracker in trackers.items():
            print("Computing tracking with", tracker_name)
            start = time.time()
            predictions = track(path, init_tracker(gt), tracker)
            accuracy, robustness = evaluate(predictions, gt)
            duration = time.time() - start
            print(f'accuracy = {accuracy}, robustness = {robustness}')
            f = open(f'{output_folder}/result.txt', 'a+')
            f.write('{};{};{:.2f};{:.2f};{:.3f}\n'.format(dataset, tracker_name, accuracy * 100.0, robustness * 100.0, duration))
            f.close()
            f = open(f'{output_folder}/{dataset}_{tracker_name}.txt', 'a+')
            for bb in predictions:
                f.write(f'{bb}\n')
            f.close()


if __name__ == '__main__':
    main()
