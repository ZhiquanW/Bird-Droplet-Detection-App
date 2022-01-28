import copy
from multiprocessing.sharedctypes import Value
import os
from typing import ItemsView
import dearpygui.dearpygui as dpg
from collections import namedtuple

from matplotlib.pyplot import semilogx, show
from numpy import core
import torch
from torch._C import dtype
import dpg_utils
import numpy as np
import utils
from tags import *
import csv

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])


def image_selector_callback(sender, user_data, app):
    # clear previous images  ------别名已存在
    dpg_utils.clear_dtextures(app.img_dtexture_list)
    dpg_utils.clear_dtextures(app.detection_notation_list)
    # dpg_utils.clear_drawlist(item_tags.texture_tags)
    dpg_utils.clear_drawlist(item_tags.detection_tags)
    dpg_utils.clear_drawlist([item_tags.target_ara_texture])
    app.blue_offset = [0, 0]
    app.target_area_bottom_right = [0, 0]
    app.target_area_top_left = [0, 0]
    app.detection_data = [[], [], [], [], []]
    app.droplet_dict_locs = {
        "Type One": [],
        "Type Two": [],
        "Type Three": [],
        "Type Four": [],
        "Type Five": [],
    }
    app.droplet_dict_num = {
        "Type One": [],
        "Type Two": [],
        "Type Three": [],
        "Type Four": [],
        "Type Five": [],
    }
    img_keys = []
    img_types = []
    img_path = []
    for img_name in user_data["selections"].keys():
        img_path.append(user_data["selections"][img_name])
        name_features = str(img_name).split(".")[0].split("_")
        img_types.append(name_features[-1].lower())
        img_keys.append("_".join(name_features[:-1]))
    if img_keys.count(img_keys[0]) != len(img_keys):
        print("two images does not have the same key: {keys}".format(img_keys))
        return
    if not ("bf" in img_types and "e" in img_types):
        print(
            "the type of the two images should be 'bf' or 'e': {types}".format(
                types=img_types
            )
        )
        return
    if img_types[0].lower() == "bf":
        img_path_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0].lower() == "e":
        img_path_pair = ImgPathPair(bright=img_path[1], blue=img_path[0])

    print(
        "record target image paths: \n\t1)[bright] {br_img_name}\n\t2)[blue] {bl_img_name}".format(
            br_img_name=img_path_pair.bright, bl_img_name=img_path_pair.blue
        )
    )

    # set new image pair path
    app.img_pair = img_path_pair
    # update text in main panel
    dpg.set_value(
        "main_panel_bright_img_id",
        value="bright image: {img_name}".format(
            img_name=img_path_pair.bright.split("/")[-1],
        ),
    )
    dpg.set_value(
        "main_panel_blue_img_id",
        value="blue image: {img_name}".format(
            img_name=img_path_pair.blue.split("/")[-1],
        ),
    )
    # bright image cell
    # br_img_cell = dpg_utils.add_texture_to_workspace(
    # img_path_pair.bright, item_tags.texture_tags[0], app.yaxis, True
    # )
    br_dtexture: dpg_utils.drawable_texture = dpg_utils.add_img_texture_to_workspace(
        img_path_pair.bright, item_tags.texture_tags[0], app.yaxis, True, True
    )
    # blue image cell
    # bl_img_cell = dpg_utils.add_texture_to_workspace(
    # img_path_pair.blue, item_tags.texture_tags[1], app.yaxis, False
    # )
    bl_dtexture = dpg_utils.add_img_texture_to_workspace(
        img_path_pair.blue, item_tags.texture_tags[1], app.yaxis, False, True
    )
    # heatmap image cell
    hm_dtexture = dpg_utils.add_notation_buff_to_workspace(
        br_dtexture.size, item_tags.texture_tags[2], app.yaxis, False, True
    )
    # 5 detection types
    for i in range(5):
        app.detection_notation_list.append(
            dpg_utils.add_notation_buff_to_workspace(
                br_dtexture.size, item_tags.detection_tags[i], app.yaxis, True, True
            )
        )
    # target area texture
    app.target_area_dtexture = dpg_utils.add_notation_buff_to_workspace(
        br_dtexture.size, item_tags.target_ara_texture, app.yaxis, True, True
    )
    app.target_area_bottom_right = br_dtexture.size
    dpg.set_value(
        app.item_tag_dict[item_tags.target_area_bottom_right_slider], br_dtexture.size
    )
    dpg.fit_axis_data(app.xaxis)
    dpg.fit_axis_data(app.yaxis)
    # add cell info to app
    app.img_dtexture_list[0] = br_dtexture
    app.img_dtexture_list[1] = bl_dtexture
    app.img_dtexture_list[2] = hm_dtexture
    # inform app that the image is loaed
    app.image_loaded = True

    # crop target area
    # enable_all_items(app)


def check_image_loaded(app):
    if not app.image_loaded:
        return False
    return True


def detect_droplets(sender, user_data, app):
    if not app.image_loaded:
        print("images are not loaded")
        return
    print("start detection on: {d}".format(d=app.target_device))
    droplet_num, predicted_map, predicted_heatmap = utils.binary_droplet_detection(
        app.img_pair.blue,
        app.img_pair.bright,
        app.batch_size,
        app.padding,
        app.stride,
        app.winsize,
        app.target_area_top_left,
        app.target_area_bottom_right,
        app.blue_offset,
        threshold=0.7,
        erosion_iter=1,
        model=app.models[app.target_type],
        device=app.target_device,
        verbose=True,
    )
    app.droplet_num = droplet_num
    print("end detection: {d}".format(d=app.droplet_num))
    # get all droplet_locs
    all_droplet_locs = utils.droplet_locs(
        predicted_map, app.img_dtexture_list[0].size[0], app.target_area_top_left
    )
    # clean_similar_locs
    print(
        "(app.target_type_names)[app.target_type]:",
        (app.target_type_names)[app.target_type],
    )
    app.droplet_dict_locs[
        (app.target_type_names)[app.target_type]
    ] = utils.clean_similar_locs(all_droplet_locs)
    print("app.droplet_dict_locs:", app.droplet_dict_locs)
    # draw rectangle
    utils.draw_detected_droplets(
        dtexture=app.detection_notation_list[app.target_type],
        droplet_locs=app.droplet_dict_locs[app.target_type_names[app.target_type]],
        rect_color=app.droplet_dict_colors[app.target_type_names[app.target_type]],
        rectangle_size=app.rectangle_size,
    )
 
    # set heatmap   ------闪退
    # dpg_utils.set_heatmap(predicted_heatmap)
    # setting rect
    # app.setting_rect_group()
    # enable_all_rect_items(app)
    # dpg.show_item("setting rect group")


def update_blue_offset(sender, user_data, app):
    if not check_image_loaded(app):
        return
    app.blue_offset[0] = user_data[0]
    app.blue_offset[1] = user_data[1]
    # print(app.blue_offset)
    dpg.configure_item(
        app.img_dtexture_list[1].image_series_tag,
        bounds_min=app.img_dtexture_list[1].top_left + app.blue_offset,
        bounds_max=app.img_dtexture_list[1].bottom_right + app.blue_offset,
    )


def select_display_raw_texture(sender, user_data, app):
    if not check_image_loaded(app):
        return
    texture_tag = user_data
    texture_idx = item_tags.texture_tags.index(texture_tag)
    app.display_raw_texture_type = texture_idx
    # disable all textures:
    for i in range(len(app.img_dtexture_list)):
        dpg.configure_item(app.img_dtexture_list[i].image_series_tag, show=False)
    # enable target texture
    dpg.configure_item(app.img_dtexture_list[texture_idx].image_series_tag, show=True)


def switch_display_raw_texture(sender, user_data, app):
    app.display_raw_texture_type += 1
    app.display_raw_texture_type %= len(app.img_dtexture_list) - 1
    # disable all textures:
    for i in range(len(app.img_dtexture_list)):
        dpg.configure_item(app.img_dtexture_list[i].image_series_tag, show=False)
    # enable target texture
    dpg.configure_item(
        app.img_dtexture_list[app.display_raw_texture_type].image_series_tag, show=True
    )


def update_padding(sender, user_data, app):
    app.padding = user_data


def update_stride(sender, user_data, app):
    app.stride = user_data


def update_win_size(sender, user_data, app):
    app.winsize = user_data


def swtich_target_type(sender, user_data, app):
    # names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
    # target_type = names.index(user_data)
    target_type = app.target_type_names.index(user_data)
    app.target_type = target_type
    # print("ctarget type: {d}".format(d=names[app.target_type]))
    print("ctarget type: {d}".format(d=app.target_type_names[app.target_type]))


def set_device(sender, user_data, app):
    if user_data == "cpu":
        app.target_device = torch.device("cpu")
    elif user_data == "gpu":
        print("cuda available: {d}".format(d=torch.cuda.is_available()))
        if torch.cuda.is_available():
            app.target_device = torch.device("cuda")
        else:
            dpg.set_value(app.item_tag_dict[item_tags.device_selector], "cpu")
    print("current device: {d}".format(d=app.target_device))


def enable_all_items(app):
    for key, val in app.item_tag_dict.items():
        dpg.enable_item(val)


def enable_all_rect_items(app):
    for key, val in app.rect_item_tag_dict.items():
        dpg.enable_item(val)


# def add_droplet_manually(sender, user_data, app: app):
#     if dpg.is_item_hovered(item_tags.image_plot_workspace):
#         mouse_pos = np.array(dpg.get_plot_mouse_pos(), dtype=np.integer)
#         app.detection_data[app.target_type].append(mouse_pos)
#         dpg_utils.update_detection_result(app)
#         # print(app.detection_data)


def set_rect_size(sender, user_data, app):
    app.rectangle_size = user_data
    utils.draw_detected_droplets(
        dtexture=app.detection_notation_list[app.target_type],
        droplet_locs=app.droplet_dict_locs[app.target_type_names[app.target_type]],
        rect_color=app.droplet_dict_colors[app.target_type_names[app.target_type]],
        rectangle_size=app.rectangle_size,
    )
    return app.rectangle_size


def rect_color(sender, user_data, app):
    new_rect_color = tuple((np.array(user_data) * 255.0).astype(np.uint8))
    # add color_tuple to the droplet_dict_colors
    app.droplet_dict_colors[app.target_type_names[app.target_type]] = new_rect_color
    utils.draw_detected_droplets(
        dtexture=app.detection_notation_list[app.target_type],
        droplet_locs=app.droplet_dict_locs[app.target_type_names[app.target_type]],
        rect_color=app.droplet_dict_colors[app.target_type_names[app.target_type]],
        rectangle_size=app.rectangle_size,
    )


def switch_droplet_manual_detectio_mode(sender, user_data, app):
    # print(app.enable_manual_detection_mode)
    app.enable_manual_detection_mode = not app.enable_manual_detection_mode
    dpg.set_value(
        app.item_tag_dict[item_tags.maunal_mode_radio], app.enable_manual_detection_mode
    )


def operate_droplet_manually(sender, user_data, app):
    # check if maunal detection mode is enabled
    if app.enable_manual_detection_mode:
        if dpg.is_item_hovered(item_tags.image_plot_workspace):
            # get droplet loc
            mouse_loc = dpg.get_plot_mouse_pos()
            if dpg.is_key_down(dpg.mvKey_Control):
                print("delete buttom")
                nearest_droplet_id = utils.find_nearest_droplet_by_type(
                    np.array(mouse_loc, dtype=np.int),
                    app.droplet_dict_locs[app.target_type_names[app.target_type]],
                    app.rectangle_size,
                )
                if nearest_droplet_id > -1:
                    app.droplet_dict_locs[app.target_type_names[app.target_type]].pop(
                        nearest_droplet_id
                    )
                    utils.draw_detected_droplets(
                        dtexture=app.detection_notation_list[app.target_type],
                        droplet_locs=app.droplet_dict_locs[
                            app.target_type_names[app.target_type]
                        ],
                        rect_color=app.droplet_dict_colors[
                            (app.target_type_names)[app.target_type]
                        ],
                        rectangle_size=app.rectangle_size,
                    )
            else:
                # add loc to the droplet_dict_locs
                app.droplet_dict_locs[app.target_type_names[app.target_type]].append(
                    np.array(mouse_loc, dtype=np.int)
                )
                utils.draw_detected_droplets(
                    app.detection_notation_list[app.target_type],
                    droplet_locs=app.droplet_dict_locs[
                        app.target_type_names[app.target_type]
                    ],
                    rect_color=app.droplet_dict_colors[
                        app.target_type_names[app.target_type]
                    ],
                    rectangle_size=app.rectangle_size,
                )
                print("app.droplet_dict_locs:", app.droplet_dict_locs)
        else:
            # print("Outside the plot")
            pass


def update_target_area_top_left(sender, user_data, app):
    if not check_image_loaded(app):
        return
    app.target_area_top_left = [user_data[0], user_data[1]]
    dpg_utils.draw_target_area_dtexture(
        app.target_area_dtexture,
        app.target_area_top_left[0],
        app.target_area_top_left[1],
        app.target_area_bottom_right[0],
        app.target_area_bottom_right[1],
    )


def update_target_area_bottom_right(sender, user_data, app):
    if not check_image_loaded(app):
        return
    app.target_area_bottom_right = [user_data[0], user_data[1]]
    dpg_utils.draw_target_area_dtexture(
        app.target_area_dtexture,
        app.target_area_top_left[0],
        app.target_area_top_left[1],
        app.target_area_bottom_right[0],
        app.target_area_bottom_right[1],
    )


def crop_target_area(sender, user_data, app):
    h0, h1, w0, w1 = utils.crop_rg_image(app.img_pair.bright)
    app.target_area_bottom_right = [h1, w1]
    app.target_area_top_left = [h0, w0]
    dpg.set_value(app.item_tag_dict[item_tags.target_area_top_left_slider], [h0, w0])
    dpg.set_value(
        app.item_tag_dict[item_tags.target_area_bottom_right_slider], [h1, w1]
    )
    dpg_utils.draw_target_area_dtexture(app.target_area_dtexture, h0, w0, h1, w1)


def export_location(sender, user_data, app):
    try:
        import json 
        file_name = app.img_pair.bright.split(".")[0]+".json"
        np2list_dict = copy.deepcopy(app.droplet_dict_locs)
        for key in np2list_dict.keys():
            for i in range(len(np2list_dict[key])):
                np2list_dict[key][i] = np2list_dict[key][i].tolist()
        with open(file_name, 'w') as fp:
            json.dump(np2list_dict, fp,  indent=4)
        print("export location file: {fn}".format(file_name))
    except Exception as e: 
        print("[error] export_location: {emsg}".format(emsg = e))


def export_density_data(sender, user_data, app):
    # if os.path.exists(app.export_file_path):
    with open(app.density_file_path, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        br_img_name = app.img_pair.bright.split("/")[-1]
        br_img_feats = br_img_name.split("_")
        site_num = br_img_feats[-2]
        droplets_num = [len(v) for v in app.droplet_dict_locs.values()]
        csv_data_row = [br_img_name, site_num, np.sum(droplets_num)] + droplets_num
        csv_writer.writerow(csv_data_row)


def set_density_data_file(sender, user_data, app):
    app.density_file_path = user_data["file_path_name"]
    dpg.set_value(item_tags.export_path_txt, user_data["file_name"])
    print(user_data)


def export_distances_data(sender, user_data, app):
    rows = []
    with open(app.distance_file_path, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        br_img_name = app.img_pair.bright.split("/")[-1]
        br_img_feats = br_img_name.split("_")
        site_num = br_img_feats[-2]
        for k, v in app.droplet_dict_locs.items():
            for loc in v:
                row = [br_img_name, site_num, "oil droplet", k, loc[0], loc[1]]
                rows.append(row)
        csv_writer.writerows(rows)


def debug_callbacks(sender, user_data, app):
    print("A")
