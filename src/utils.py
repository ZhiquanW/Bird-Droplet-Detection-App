from typing import Dict, List, Tuple
from matplotlib import image
from numpy import core
from numpy.lib.type_check import imag
import plotly.graph_objects as go
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import ImageOps
import torchvision
import tqdm
import cv2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion
from plotly.subplots import make_subplots
import os
import json
import core
import random
import math
from PIL import Image, ImageOps
import dearpygui.dearpygui as dpg
from PIL import Image, ImageOps,ImageDraw

def format_dict_str(src_dict):
    str_dict = ""
    for key, val in src_dict.items():
        str_dict += str(key) + ": " + str(val) + "\n\n"
    return str_dict


def log_params(params_dict):
    for key, val in params_dict.items():
        print(key, ": ", val)


def pred_info(preds, labels):
    return {"correct_num": preds.argmax(dim=1).eq(labels).sum().item()}


def confusion_mat(preds: torch.Tensor, labels: torch.Tensor, class_num: int):
    preds = preds.cpu()
    labels = labels.cpu()
    tmp_cm = np.zeros((class_num, class_num))
    for i in range(len(labels)):
        tmp_cm[int(labels[i].numpy())][int(preds[i].numpy())] += 1.0
    return tmp_cm


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100, 2)

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig


def train_iteration_log(
    epoch: int, iter: int, loss: float, correct_num: int, sample_num: int
):
    print("Epoch: {e} - Iteration: {i}, loss = {l:.6f}".format(e=epoch, i=iter, l=loss))
    print(
        "correct: {cn}/{tt}: {p:.2f} %".format(
            cn=correct_num, tt=sample_num, p=correct_num / sample_num * 100
        )
    )


def train_epoch_log(epoch: int, loss: float, correct_num: int, sample_num: int):
    print(
        "{pre_w}== train: {e} =={post_w}".format(
            e=epoch, pre_w="#" * 30, post_w="#" * 30
        )
    )
    print("loss = {l:.4f}".format(l=loss))
    print(
        "correct: {cn}/{tt}: {p:.4f} %".format(
            cn=correct_num, tt=sample_num, p=correct_num / sample_num * 100,
        )
    )


def validation_epoch_log(epoch: int, correct_num: int, sample_num: int):
    print(
        "{pre_w}== validation: {e} =={post_w}".format(
            e=epoch, pre_w="#" * 30, post_w="#" * 30
        )
    )
    print(
        "correct: {cn}/{tt}: {p:.4f} %".format(
            cn=correct_num, tt=sample_num, p=correct_num / sample_num * 100,
        )
    )

    # self.record_board.add_figure(
    #     "t confusion matrix on test {e}".format(e=e),
    #     utils.plot_confusion_matrix(
    #         cm=confusion_matrix, classes=["0", "1", "2", "3", "4", "5"]
    #     ),
    # )


def train_tensorboard_log(
    record_board,
    epoch: int,
    confusion_mat: np.ndarray,
    loss: float,
    correct_num: int,
    sample_num: int,
    learning_rate: float,
):
    if epoch % 100 == 0:
        record_board.add_figure(
            "confusion matrix on train {e}".format(e=epoch),
            plot_confusion_matrix(
                cm=confusion_mat, classes=["0", "1", "2", "3", "4", "5"], normalize=True
            ),
        )

    record_board.add_scalar("loss", loss, epoch)
    record_board.add_scalar("correct_num", correct_num, epoch)
    record_board.add_scalar("accuracy", correct_num / sample_num * 100.0, epoch)
    # record_board.add_scalar("learning rate", learning_rate, epoch)


def validation_tensorboard_log(
    record_board,
    epoch: int,
    confusion_mat: np.ndarray,
    correct_num: int,
    sample_num: int,
):
    if epoch % 10 == 0:
        record_board.add_figure(
            "confusion matrix on validation {e}".format(e=epoch),
            plot_confusion_matrix(
                cm=confusion_mat, classes=["0", "1", "2", "3", "4", "5"], normalize=True
            ),
        )
    record_board.add_scalar("correct_num", correct_num, epoch)
    record_board.add_scalar("accuracy", correct_num / sample_num * 100.0, epoch)


# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel."""

#     x = np.linspace(-nsig, nsig, kernlen + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kern2d = np.outer(kern1d, kern1d)
#     return kern2d / kern2d.sum()


def predict_droplet_densitymap(
    droplet_imgs: Tuple,
    model,
    batch_size:int = 128,
    padding: int = 7,
    stride: int = 2,
    win_size=30,
    verbose: bool = False,
    device=torch.device("cpu"),
):
    target_device = device
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
    )
    img_dataset = core.SlidingWindowDataset(
        images=droplet_imgs,
        padding=padding,
        win_size=win_size,
        stride=stride,
        transform=image_transform,
    )
    img_loader = torch.utils.data.DataLoader(
        dataset=img_dataset, batch_size=batch_size, num_workers=12,
    )
    # detection_heat_map = np.zeros((7, IMG_H, IMG_W))
    # detection_heat_map[-1, :, :] = 1
    model.to(target_device)
    model.eval()
    scores_list = []
    with torch.no_grad():
        for t, (images, _) in enumerate(tqdm.tqdm(img_loader, disable=not verbose)):
            # move to device, e.g. GPU
            images = images.to(device=target_device, dtype=torch.float32,)
            scores = model(images)
            scores_list.append(scores)
    combined_scores = torch.cat(scores_list, dim=0)
    combined_scores /= torch.sum(combined_scores, 1, keepdim=True)
    return (
        combined_scores.reshape((img_dataset.MAP_H, img_dataset.MAP_W, 2))
        .cpu()
        .numpy()
        .transpose(2, 0, 1)
    )


def densitymap2heatmap(img_size, density_layer, stride, winsize, padding):
    IMG_W, IMG_H, = img_size
    MAP_H, MAP_W = density_layer.shape
    detection_heatmap = np.zeros((IMG_H + 2 * padding, IMG_W + 2 * padding))
    for h in range(MAP_H):
        for w in range(MAP_W):
            top = h * stride
            bottom = top + winsize
            left = w * stride
            right = left + winsize
            detection_heatmap[top:bottom, left:right] += density_layer[h, w]
    return detection_heatmap


def detection_img_heatmap(
    droplet_imgs: Tuple,
    model,
    padding: int = 7,
    stride: int = 2,
    win_size=30,
    verbose: bool = False,
    gpu=False,
):

    target_device = torch.device("cuda") if gpu else torch.device("cpu")
    e_img, bf_img = droplet_imgs
    IMG_H, IMG_W = e_img.size
    padded_e_image = ImageOps.expand(e_img, (padding, padding, padding, padding))
    padded_bf_image = ImageOps.expand(bf_img, (padding, padding, padding, padding))
    # build heat map
    MAP_H, MAP_W = (
        (np.array([IMG_H, IMG_W]) + 2 * padding - win_size) / stride + 1
    ).astype(int)
    detection_heat_map = np.zeros((7, IMG_H, IMG_W))
    detection_heat_map[-1, :, :] = 1
    model.to(target_device)
    model.eval()
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
    )

    with torch.no_grad():
        for h in tqdm.trange(MAP_H, desc="detecting droplets", disable=not verbose):
            for w in range(MAP_W):
                top = h * stride
                bottom = top + win_size
                left = w * stride
                right = left + win_size
                e_slided = padded_e_image.crop((left, top, right, bottom))
                bf_slided = padded_bf_image.crop((left, top, right, bottom))
                e_transformed = image_transform(e_slided)
                bf_transformed = image_transform(bf_slided)
                image_mat = torch.cat((e_transformed, bf_transformed), 0)
                input_img = image_mat.to(
                    device=target_device, dtype=torch.float32
                ).unsqueeze(0)
                # print(input_img.shape)
                scores = model(input_img)
                prods = torch.nn.functional.softmax(scores[0], dim=0)
                _, pred = scores.max(1)
                detection_heat_map[:-1, top:bottom, left:right] += prods.reshape(
                    6, 1, 1
                ).numpy()
                detection_heat_map[-1, top:bottom, left:right] += 1
    detection_heat_map[:-1, :, :] /= detection_heat_map[-1, :, :]
    return detection_heat_map


def labeled_img_heatmap(
    img_size: Tuple,
    map_size: Tuple,
    droplet_info: Dict,
    padding: int = 7,
    stride: int = 2,
    win_size=30,
):
    MAP_H, MAP_W = map_size
    IMG_H, IMG_W = img_size
    labeled_map = np.zeros((IMG_H + 2 * padding, IMG_W + 2 * padding)) + 5
    labeled_heat_map = np.zeros((7, IMG_H, IMG_W))
    labeled_heat_map[-1, :, :] = 1
    for i in range(5):
        type_key = "type{type_id}".format(type_id=i + 1)
        for loc in droplet_info[type_key]["loc"]:
            x, y = np.array(loc) + padding
            labeled_map[x, y] = i
    for h in range(MAP_H):
        for w in range(MAP_W):
            for i in range(5):
                top = h * stride
                bottom = top + win_size
                left = w * stride
                right = left + win_size
                if i in labeled_map[top:bottom, left:right]:
                    labeled_heat_map[i, top:bottom, left:right] += 1
            labeled_heat_map[-1, top:bottom, left:right] += 1
    # for h in range(MAP_H):
    #     for w in range(MAP_W):
    #         if labeled_heat_map[-1, h, w] > 0:
    #             labeled_heat_map[:-1, h, w] /= labeled_heat_map[-1, h, w]
    # labeled_heat_map[:-1, :, :] /= labeled_heat_map[-1, :, :]
    return labeled_heat_map, labeled_map


def predicted_img_heatmap(
    img_size: Tuple, map_size: Tuple, densitymap: np.ndarray, stride: int, win_size: int
):
    MAP_H, MAP_W = map_size
    IMG_H, IMG_W = img_size
    predicted_heatmap = np.zeros((6, IMG_H, IMG_W))
    for h in range(MAP_H):
        for w in range(MAP_W):
            top = h * stride
            bottom = top + win_size
            left = w * stride
            right = left + win_size
            predicted_heatmap[:, top:bottom, left:right] += densitymap[:, h, w].reshape(
                6, 1, 1
            )
    return predicted_heatmap


def droplet_loc_filter(target_map, threshold=0.7, erosion_iter: int = 2):
    min_prob = np.min(target_map)
    max_prob = np.max(target_map)
    normalized_map = (target_map - min_prob) / (max_prob - min_prob)
    mean_prob = np.mean(normalized_map)
    filter_idx_below_mean = normalized_map < mean_prob
    normalized_map[filter_idx_below_mean] = 0
    filter_idx_below_threshold = normalized_map < 0.7
    normalized_map[filter_idx_below_threshold] = 0
    # filter_idx_above_threshold = normalized_map > 0.8
    # normalized_map[filter_idx_above_threshold] = 1

    erosion_kernel = np.ones((3, 3), np.uint8)
    normalized_map = cv2.erode(normalized_map, erosion_kernel, iterations=erosion_iter)

    def detect_peaks(image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        # define an 8-connected neighborhood
        # neighborhood = generate_binary_structure(2,2)
        neighborhood = np.ones((25, 25))

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = image == 0

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1
        )

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background
        return detected_peaks

    detected_droplets = np.array(detect_peaks(normalized_map)).astype(int)
    return detected_droplets


# methods module
def draw_droplet(img_bgr, droplet_info, half_size, types: List = [1, 1, 1, 1, 1]):
    type_colors = [
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [255, 0, 255],
        [0, 200, 255],
    ]  # bgr
    padded_img = cv2.copyMakeBorder(
        img_bgr, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT,
    )
    for i in range(5):
        if types[i] == 0:
            continue
        type_key = "type{type_id}".format(type_id=i + 1)
        for loc in droplet_info[type_key]["loc"]:
            # if i == 0 :
            #     print(loc)
            loc_xy = np.array(loc)
            top_left = loc_xy - half_size + half_size
            bottom_right = loc_xy + half_size + half_size
            h0, w0 = top_left
            h1, w1 = bottom_right
            padded_img = cv2.rectangle(
                padded_img, (w0, h0), (w1, h1), type_colors[i], 2
            )
            # padded_img[loc[0] + half_size * 2, loc[1] + half_size * 2] = [255, 255, 255]
    return padded_img


def draw_predicted_on_raw(
    image_folder, b_image_name, predicted_map, draw_winsize=10, padding=7
):
    b_image_bgr = cv2.imread(image_folder + b_image_name)
    padded_image_bgr = cv2.copyMakeBorder(
        b_image_bgr, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
    )

    IMG_H, IMG_W, _ = padded_image_bgr.shape
    assert (IMG_H, IMG_W) == predicted_map.shape
    for h in range(IMG_H):
        for w in range(IMG_W):
            if predicted_map[h, w] == 1:
                top = h - draw_winsize
                bottom = h + draw_winsize
                left = w - draw_winsize
                right = w + draw_winsize
                padded_image_bgr = cv2.rectangle(
                    padded_image_bgr, (left, top), (right, bottom), [255, 0, 0], 2
                )
                padded_image_bgr[h, w] = [255, 255, 255]
    return padded_image_bgr


def normalize_map(in_map):
    max_val = np.max(in_map)
    min_val = np.min(in_map)
    return (in_map - min_val) / (max_val - min_val)


def count_droplets(detected_droplets_map):
    MAP_H, MAP_W = detected_droplets_map.shape
    visited_map = np.zeros_like(detected_droplets_map)

    def dfs_count(loc: np.ndarray):
        # print(loc)
        search_dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        x, y = loc
        if visited_map[x][y] == 1:
            return 0
        visited_map[x, y] = 1

        if detected_droplets_map[x][y] == 1:
            for dir in search_dirs:
                next_loc = loc + dir
                if (
                    next_loc[0] >= 0
                    and next_loc[0] < MAP_H
                    and next_loc[1] >= 0
                    and next_loc[1] < MAP_W
                ):
                    dfs_count(next_loc)
        else:
            return 0
        return 1

    droplet_counter = 0
    for h in range(MAP_H):
        for w in range(MAP_W):
            if visited_map[h][w] == 0:
                droplet_exist = dfs_count(np.array([h, w]))
                droplet_counter += droplet_exist

    return droplet_counter


def droplet_detection(
    droplet_imgs,
    model,
    padding: int = 7,
    stride=2,
    win_size=30,
    verbose: bool = False,
    device=torch.device("cpu"),
):
    e_image, bf_image = droplet_imgs
    result_dict = {}
    detection_heatmap = predict_droplet_densitymap(
        (e_image, bf_image),
        model,
        padding=padding,
        stride=stride,
        win_size=win_size,
        device=device,
        verbose=verbose,
    )
    for i in range(5):
        target_layer = np.array(detection_heatmap[i, :, :])
        detected_droplets = droplet_loc_filter(target_layer)
        d_counter = count_droplets(detected_droplets)
        result_dict[str(i)] = (d_counter, detected_droplets)
    return result_dict, detection_heatmap


def binary_droplet_detection(
    e_image_name,
    b_image_name,
    batch_size,
    padding,
    stride,
    winsize,
    threshold,
    erosion_iter,
    model,
    device,
    disable_border=20,
    verbose=False,
):
    e_image = Image.open(e_image_name)
    droplet_densitymap = binary_droplet_detection_heatmap(
        e_image_name,
        b_image_name,
        batch_size,
        padding,
        stride,
        winsize,
        model,
        device,
        verbose=verbose,
    )
    droplet_heatmap = densitymap2heatmap(
        e_image.size, droplet_densitymap[1, :, :], stride, winsize, padding
    )
    droplet_heatmap[:, :disable_border,] = 0
    droplet_heatmap[-disable_border:, :] = 0
    predicted_map = droplet_loc_filter(droplet_heatmap, threshold, erosion_iter)
    droplet_num = count_droplets(predicted_map)
    return droplet_num, predicted_map, droplet_heatmap


def check_binary_result(
    img_folder,
    b_image_name,
    target_type,
    droplet_info,
    target_droplet_num,
    predicted_map,
    padding,
    show: False,
):
    type_key = "type{type_id}".format(type_id=target_type + 1)
    labeled_num = droplet_info[type_key]["num"]
    offset_num = np.abs(labeled_num - target_droplet_num)
    predicted_raw_img = draw_predicted_on_raw(
        img_folder, b_image_name, predicted_map, padding=padding
    )
    return offset_num, predicted_raw_img if show else None


def show_original_images(e_img, bf_img):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(e_img)
    ax[1].imshow(bf_img)
    plt.show()


def show_labeled_images(
    img_path, b_img_name, droplet_info, win_size=10, targets=[1, 1, 1, 1, 1], show=True
):
    imgs = []
    for i in range(5):
        if targets[i] != 1:
            continue
        type_key = "type{type_id}".format(type_id=i + 1)
        b_bgr = cv2.imread(img_path + b_img_name)
        sample_b_bgr = draw_droplet(b_bgr, droplet_info, win_size, targets)
        sample_b_rgb = cv2.cvtColor(sample_b_bgr, cv2.COLOR_BGR2RGB)
        imgs.append((i, sample_b_rgb))

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs))
        for idx, img in imgs:
            target_ax = ax if len(imgs) == 1 else ax[i]
            target_ax.set_title(type_key + ": " + str(idx + 0))
            target_ax.imshow(sample_b_rgb)
        plt.show()
    return list(zip(*imgs))[1]


def draw_predicted_loc_map(res_dict):
    fig = make_subplots(rows=1, cols=5, shared_yaxes=False)
    counter = 0
    for k, val in res_dict.items():
        fig.add_trace(go.Heatmap(z=val[1]), row=1, col=counter + 1)
        counter += 1
    fig.update_layout(height=400, width=1000, coloraxis={"colorscale": "viridis"})
    fig.show()


def result_plot(result_dict, predicted_heatmap, labeled_map, labeled_heatmap):
    fig = make_subplots(rows=4, cols=5)
    for i in range(5):
        # type_key = "type{type_id}".format(type_id=i + 1)
        type_filter = labeled_map == i
        human_label_map = np.zeros_like(labeled_map)
        human_label_map[type_filter] = 1
        fig.add_trace(
            go.Heatmap(
                z=normalize_map(predicted_heatmap[i, :, :]), coloraxis="coloraxis"
            ),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Heatmap(z=labeled_heatmap[i], coloraxis="coloraxis"), row=2, col=i + 1,
        )
        fig.add_trace(
            go.Heatmap(z=result_dict[str(i)][1], coloraxis="coloraxis"),
            row=3,
            col=i + 1,
        )
        fig.add_trace(
            go.Heatmap(z=human_label_map, coloraxis="coloraxis"), row=4, col=i + 1,
        )
    fig.update_layout(height=1200, width=1500, coloraxis={"colorscale": "viridis"})
    fig.show()


def collect_images(raw_image_dir, label_dir):
    # construct pairs of images (raw_e,raw_bf) in test image folders + label file
    _, _, raw_image_names = next(os.walk(raw_image_dir))
    _, _, label_names = next(os.walk(label_dir))
    # print('find {n} counted images'.format(n = len(raw_image_names)))
    # print('find {n} label files'.format(n = len(label_names)))
    pair_dict = {}
    for raw_image_name in raw_image_names:
        image_name = raw_image_name.split(".")[0]
        key_name = "_".join(image_name.split("_")[:-1])
        image_type = 0 if image_name[-1] == "e" else 1
        if key_name in pair_dict.keys():
            pair_dict[key_name][image_type] = raw_image_name
            pair_dict[key_name][3] += 1
        else:
            pair_dict[key_name] = [[], [], [], 0]
            pair_dict[key_name][image_type] = raw_image_name
            pair_dict[key_name][3] += 1
    # pair label and load droplet location information
    for label_name in label_names:
        label_key = label_name.split(".")[0]
        assert label_key in pair_dict
        with open(label_dir + label_name) as f:
            droplet_loc = json.load(f)
        pair_dict[label_key][2] = droplet_loc
        pair_dict[label_key][3] += 1
    return pair_dict
    # # check image - label pairs

    # paired_counter = 0
    # for key,val in pair_dict.items():
    #     if val[3] == 3:
    #         paired_counter += 1
    # print('paired {n} raw images'.format(n = paired_counter))


def check_result(result_dict, droplet_info):
    check_info = {}
    for i in range(5):
        type_key = "type{type_id}".format(type_id=i + 1)
        detected_num = result_dict[int(i)][0]
        labeled_num = droplet_info[type_key][["num"]]
        accuracy = np.abs(detected_num - labeled_num) / labeled_num
        check_info[int(i)] = [detected_num, labeled_num, accuracy]
    return check_info


def analysis_predicted_result(droplet_info, result_dict):
    labeled_num = np.array([v["num"] for v in droplet_info.values()])
    predicted_num = np.array([v[0] for v in result_dict.values()])
    offset_num = labeled_num - predicted_num
    error_rate = np.abs(offset_num) / labeled_num
    return offset_num, error_rate


def label_raw_map(height, width, droplet_info, half_size):
    # generate a map (5 layers)with the droplet locations is labeled with 1
    label_map = np.zeros((5, height, width))
    for i in range(5):
        type_key = "type{type_id}".format(type_id=i + 1)
        for loc in droplet_info[type_key]["loc"]:
            loc_xy = np.array(loc)
            top_left = loc_xy - half_size + half_size * 2
            bottom_right = loc_xy + half_size + half_size * 2
            top_left[top_left < 0] = 0
            bottom_right[bottom_right < 0] = 0
            label_map[i][
                top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]
            ] = 1
    return label_map


def droplet_cut_plus(
    img_key,
    img_folder,
    e_image_name,
    b_image_name,
    droplet_info,
    label_half_size,
    cut_half_size_range,
    threshold,
    target_folder,
):
    e_image = cv2.imread(img_folder + e_image_name)
    b_image = cv2.imread(img_folder + b_image_name)
    padded_b_img = cv2.copyMakeBorder(
        b_image,
        label_half_size * 2,
        label_half_size * 2,
        label_half_size * 2,
        label_half_size * 2,
        cv2.BORDER_CONSTANT,
    )
    padded_e_img = cv2.copyMakeBorder(
        e_image,
        label_half_size * 2,
        label_half_size * 2,
        label_half_size * 2,
        label_half_size * 2,
        cv2.BORDER_CONSTANT,
    )

    h, w, _ = padded_b_img.shape
    label_map = label_raw_map(h, w, droplet_info, label_half_size)
    # label_map[1] = 1
    _, H, W = label_map.shape
    res_dict = {}
    res_dict[5] = []  # background type
    for i in range(5):
        res_dict[i] = []
        for h in range(0, H, 2):
            for w in range(0, W, 2):
                cut_half_size = random.randint(
                    cut_half_size_range[0], cut_half_size_range[1]
                )

                top = h
                bottom = h + cut_half_size * 2
                left = w
                right = w + cut_half_size * 2
                bottom = np.clip(bottom, 0, H)
                right = np.clip(right, 0, W)
                droplet_pixels = np.sum(label_map[i][top:bottom, left:right])
                background_pixels = np.sum(label_map[:, top:bottom, left:right])
                if droplet_pixels >= threshold:
                    res_dict[i].append(
                        (
                            padded_e_img[top:bottom, left:right],
                            padded_b_img[top:bottom, left:right],
                            label_map[i][top:bottom, left:right],
                        )
                    )
                elif (
                    background_pixels == 0
                    and top > cut_half_size * 2
                    and left > cut_half_size * 2
                    and bottom - top == cut_half_size * 2
                    and right - left == cut_half_size * 2
                ):
                    res_dict[5].append(
                        (
                            padded_e_img[top:bottom, left:right],
                            padded_b_img[top:bottom, left:right],
                            np.sum(label_map[:, top:bottom, left:right], axis=0),
                        )
                    )

    droplet_num = np.zeros(6)
    for key, val in res_dict.items():
        droplet_num[int(key)] = len(val)
    background_num = min(int(droplet_num[-1]), int(np.max(droplet_num[:-1])))
    sub_img_counter = 0
    for key, val in res_dict.items():
        if key == 5:
            image_list = random.sample(val, k=background_num)
        else:
            image_list = val
        for img in image_list:
            e_img_name = "{id}_{droplet_type}_{img_type}_{counter}_{half_size}_{threshold}.png".format(
                id=img_key,
                img_type="e",
                counter=str(sub_img_counter),
                droplet_type=key,
                half_size=cut_half_size,
                threshold=threshold,
            )
            cv2.imwrite(target_folder + e_img_name, img[0])
            b_img_name = "{id}_{droplet_type}_{img_type}_{counter}_{half_size}_{threshold}.png".format(
                id=img_key,
                img_type="b",
                counter=str(sub_img_counter),
                droplet_type=key,
                half_size=cut_half_size,
                threshold=threshold,
            )
            cv2.imwrite(target_folder + b_img_name, img[1])
            sub_img_counter += 1
    return res_dict


def binary_droplet_detection_heatmap(
    e_image_name, b_image_name, batch_size,padding, stride, winsize, model, device, verbose=False,
):
    e_image = Image.open(e_image_name)
    bf_image = Image.open(b_image_name)

    torch.cuda.empty_cache()
    droplet_densitymap = predict_droplet_densitymap(
        (e_image, bf_image),
        model,
        batch_size=batch_size,
        padding=padding,
        stride=stride,
        win_size=winsize,
        verbose=verbose,
        device=device,
    )
    return droplet_densitymap

def droplet_locs(predicted_map,w):
    cell_bool = predicted_map > 0
    locs = np.where(cell_bool)
    x_locs = w - locs[0] + 4
    y_locs = locs[1] -10
    list_x_locs = x_locs.tolist()
    list_y_locs = y_locs.tolist()
    locs = []
    for i in range(0,len(list_x_locs)):   
        locs.append([list_y_locs[i],list_x_locs[i]])
        # print(i)
    # print(locs)
    return locs

def draw_rectangle(buff_data,texture_name,droplet_locs,rect_color,rectangle_size):
    im = Image.fromarray(np.uint8(buff_data))
    im_draw = ImageDraw.Draw(im)
    bounds = []
    i = 0
    n = 0
    # print(type(locs))
    for loc in droplet_locs:
        n+=1
        bounds.append([loc[0]+rectangle_size,loc[1]-rectangle_size])
        bounds.append([loc[0]-rectangle_size,loc[1]+rectangle_size])       
        # num
        print('darwing',n)   
        # rectangle_locs
        x1, y1 =bounds[i]
        x2, y2 =bounds[i+1]
        i+=2
        # set outline
        im_draw.rectangle((x1, y1, x2, y2), outline=rect_color, width=1)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.save('detect_img.png')
    im = os.path.join(os.getcwd(),'detect_img.png')
    width, height, channels, data = dpg.load_image(im)
    # set iamge texture value
    dpg.set_value(texture_name,data)

def clean_similar_locs(droplet_locs,defalut_size = 4):
    n = 0
    for loc in droplet_locs:
        n += 1
        print("loc",loc,n)
        pr_locs = []
        for x in range(1,defalut_size):
            for y in range(1,defalut_size):
                pr_locs.append([loc[0]-x,loc[1]])
                pr_locs.append([loc[0]+x,loc[1]])
                pr_locs.append([loc[0],loc[1]+y])
                pr_locs.append([loc[0],loc[1]-y])
                pr_locs.append([loc[0]-x,loc[1]+y])
                pr_locs.append([loc[0]+x,loc[1]+y])
                pr_locs.append([loc[0]-x,loc[1]-y])
                pr_locs.append([loc[0]+x,loc[1]-y])
        for pr_loc in pr_locs:
            if pr_loc in droplet_locs:
                droplet_locs.remove(pr_loc)   
    return droplet_locs 

def find_rect_locs(loc,locs,find_size = 6):
    for x in range(find_size):
        for y in range(find_size):
            locs.append([loc[0]-x,loc[1]])
            locs.append([loc[0]+x,loc[1]])
            locs.append([loc[0],loc[1]+y])
            locs.append([loc[0],loc[1]-y])
            locs.append([loc[0]-x,loc[0]+y])
            locs.append([loc[0]+x,loc[1]+y])
            locs.append([loc[0]-x,loc[1]-y])
            locs.append([loc[0]+x,loc[0]-y])
    return locs