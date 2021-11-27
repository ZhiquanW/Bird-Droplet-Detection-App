import dearpygui.dearpygui as dpg
from collections import namedtuple

from matplotlib.pyplot import semilogx, show
from numpy import core
import torch
import dpg_utils
import numpy as np
import utils
from tags import *
from core import app

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])


def image_selector_callback(sender, app_data, app: app):
    # clear previous images  ------别名已存在
    dpg_utils.clear_drawlist(item_tags.texture_tags)
    dpg_utils.clear_drawlist(item_tags.detection_tags)
    img_keys = []
    img_types = []
    img_path = []
    for img_name in app_data["selections"].keys():
        img_path.append(app_data["selections"][img_name])
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
    if img_types[0] == "bf":
        img_path_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0] == "e":
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
    br_img_cell ,app.img_size[0] ,app.img_size[1] ,im= dpg_utils.add_texture_to_workspace(
        img_path_pair.bright, item_tags.texture_tags[0], app.yaxis, True
    )
    # blue image cell
    # bl_img_cell = dpg_utils.add_texture_to_workspace(
    # img_path_pair.blue, item_tags.texture_tags[1], app.yaxis, False
    # )
    bl_img_cell ,app.img_size[0] ,app.img_size[1] ,im= dpg_utils.add_texture_to_workspace(
        img_path_pair.blue, item_tags.texture_tags[1], app.yaxis, False
    )
    # heatmap image cell
    hm_img_cell = dpg_utils.add_image_buff_to_workspace(
        br_img_cell.size, item_tags.texture_tags[2], app.yaxis, False, True
    )
    # get buff_data
    app.buff_data = im
    # 5 detection types
    for i in range(5):
        app.detection_gallery.append(
            dpg_utils.add_image_buff_to_workspace(
                br_img_cell.size, item_tags.detection_tags[i], app.yaxis, True, True
            )
        )
    dpg.fit_axis_data(app.xaxis)
    dpg.fit_axis_data(app.yaxis)
    # add cell info to app
    app.texture_gallery.append(br_img_cell)
    app.texture_gallery.append(bl_img_cell)
    app.texture_gallery.append(hm_img_cell)
    # inform app that the image is loaed
    app.image_loaded = True
    enable_all_items(app)


def check_image_loaded(app):
    if not app.image_loaded:
        print("images are not loaded")
        return False
    return True


def detect_droplets(sender, app_data, app):
    if not check_image_loaded(app):
        return
    print("start detection: tpye{d}".format(d=app.target_device))

    droplet_num, predicted_map, predicted_heatmap = utils.binary_droplet_detection(
        app.img_pair.blue,
        app.img_pair.bright,
        app.batch_size,
        app.padding,
        app.stride,
        app.winsize,
        threshold=0.7,
        erosion_iter=1,
        model=app.models[app.target_type],
        device=app.target_device,
        verbose=True,
    )
    app.droplet_num = droplet_num
    print("end detection: {d}".format(d=app.droplet_num))
    # get all droplet_locs
    all_droplet_locs = utils.droplet_locs(predicted_map,app.img_size[0])
    # clean_similar_locs
    print("(app.target_type_names)[app.target_type]:",(app.target_type_names)[app.target_type])
    app.droplet_dict_locs[(app.target_type_names)[app.target_type]] = utils.clean_similar_locs(all_droplet_locs)
    print("app.droplet_dict_locs:",app.droplet_dict_locs)
    # draw rectangle
    utils.draw_rectangle(
        buff_data = app.buff_data,
        texture_name = (item_tags.detection_tags)[app.target_type],
        droplet_locs = app.droplet_dict_locs[(app.target_type_names)[app.target_type]],
        rect_color = app.droplet_dict_colors[(app.target_type_names)[app.target_type]],
        rectangle_size = app.rectangle_size
        )
    # set heatmap   ------闪退
    dpg_utils.set_heatmap(predicted_heatmap)
    # setting rect
    app.setting_rect_group()
    enable_all_rect_items(app)
    dpg.show_item("setting rect group")

def update_blue_offset(sender, app_data, app):
    if not check_image_loaded(app):
        return
    app.blue_offset[0] = app_data[0]
    app.blue_offset[1] = app_data[1]
    # print(app.blue_offset)
    dpg.configure_item(
        app.texture_gallery[1].image_series_tag,
        bounds_min=app.texture_gallery[1].top_left + app.blue_offset,
        bounds_max=app.texture_gallery[1].bottom_right + app.blue_offset,
    )


def switch_raw_texture(sender, app_data, app):
    if not check_image_loaded(app):
        return
    texture_tag = app_data
    texture_idx = item_tags.texture_tags.index(texture_tag)
    # disable all textures:
    for i in range(len(app.texture_gallery)):
        dpg.configure_item(app.texture_gallery[i].image_series_tag, show=False)
    # enable target texture
    dpg.configure_item(app.texture_gallery[texture_idx].image_series_tag, show=True)



def update_padding(sender, app_data, app):
    app.padding = app_data


def update_stride(sender, app_data, app):
    app.stride = app_data


def update_win_size(sender, app_data, app):
    app.winsize = app_data


def swtich_target_type(sender, app_data, app):
    # names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
    # target_type = names.index(app_data)
    print(app_data)
    target_type = app.target_type_names.index(app_data)
    app.target_type = target_type
    # print("ctarget type: {d}".format(d=names[app.target_type]))
    print("ctarget type: {d}".format(d=app.target_type_names[app.target_type]))


def set_device(sender, app_data, app):
    if app_data == "cpu":
        app.target_device = torch.device("cpu")
    elif app_data == "gpu":
        print("cuda available: {d}".format(d=torch.cuda.is_available()))
        if torch.cuda.is_available():
            app.target_device = torch.device("cuda")
    print("current device: {d}".format(d=app.target_device))


def enable_all_items(app):
    for key, val in app.item_tag_dict.items():
        dpg.enable_item(val)

def enable_all_rect_items(app):
    for key, val in app.rect_item_tag_dict.items():
        print(val)
        dpg.enable_item(val)


# def add_droplet_manually(sender, app_data, app: app):
#     if dpg.is_item_hovered(item_tags.image_plot_workspace):
#         mouse_pos = np.array(dpg.get_plot_mouse_pos(), dtype=np.integer)
#         app.detection_data[app.target_type].append(mouse_pos)
#         dpg_utils.update_detection_result(app)
#         # print(app.detection_data)

def set_rect_size(sender, app_data, app):
    app.rectangle_size = app_data
    utils.draw_rectangle(
        buff_data = app.buff_data,
        texture_name = (item_tags.detection_tags)[app.target_type],
        droplet_locs = app.droplet_dict_locs[(app.target_type_names)[app.target_type]],
        rect_color = app.droplet_dict_colors[(app.target_type_names)[app.target_type]],
        rectangle_size = app.rectangle_size,
        )
    print("set_rect_size")
    return app.rectangle_size

def rect_color(sender, app_data, app):
    new_rect_color = dpg.get_value(sender)
    color=[]
    for i in new_rect_color:
        color.append(int(i))
    color_tuple = tuple(color)   
    # print("color_tuple",color_tuple)
    # add color_tuple to the droplet_dict_colors
    app.droplet_dict_colors[(app.target_type_names)[app.target_type]] = color_tuple
    utils.draw_rectangle(
        buff_data = app.buff_data,
        texture_name = (item_tags.detection_tags)[app.target_type],
        droplet_locs = app.droplet_dict_locs[(app.target_type_names)[app.target_type]],
        rect_color = app.droplet_dict_colors[(app.target_type_names)[app.target_type]],
        rectangle_size = app.rectangle_size,
        )
    # print("rect_color")

def add_droplet_manually(sender, app_data, app):
    def add_droplet(sender,data,app):
        if dpg.is_item_hovered(item_tags.image_plot_workspace): 
            print("Add :")
            # get droplet loc 
            loc = dpg.get_plot_mouse_pos()
            list_loc = [round(loc) for loc in loc]
            # add loc to the droplet_dict_locs
            app.droplet_dict_locs[(app.target_type_names)[app.target_type]].append(list_loc)
            utils.draw_rectangle(
                buff_data = app.buff_data,
                texture_name = (item_tags.detection_tags)[app.target_type],
                droplet_locs = app.droplet_dict_locs[(app.target_type_names)[app.target_type]],
                rect_color = app.droplet_dict_colors[(app.target_type_names)[app.target_type]],
                rectangle_size = app.rectangle_size,
                )
            print("app.droplet_dict_locs:",app.droplet_dict_locs)
        else:
            print("Outside the plot")

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(
            button=0, 
            callback=add_droplet,
            user_data=app 
            )
def delete_droplet_manually(sender, app_data, app):
    def delete_droplet(sender,data,app):
        if dpg.is_item_hovered(item_tags.image_plot_workspace): 
            print("Delete :")
            # get droplet loc 
            loc = dpg.get_plot_mouse_pos()
            list_loc = [[round(loc) for loc in loc]]    # print(locs[0])    [140, 141]
            try_locs = utils.find_rect_locs(list_loc[0],list_loc,app.rectangle_size)
            # delete droplet loc
            for try_loc in try_locs:
                if try_loc in app.droplet_dict_locs[(app.target_type_names)[app.target_type]]:
                    print(app.droplet_dict_locs[(app.target_type_names)[app.target_type]])
                    app.droplet_dict_locs[(app.target_type_names)[app.target_type]].remove(try_loc)
            print("app.droplet_dict_locs:",app.droplet_dict_locs)
            utils.draw_rectangle(
                buff_data = app.buff_data,
                texture_name = (item_tags.detection_tags)[app.target_type],
                droplet_locs = app.droplet_dict_locs[(app.target_type_names)[app.target_type]],
                rect_color = app.droplet_dict_colors[(app.target_type_names)[app.target_type]],
                rectangle_size = app.rectangle_size,
                )
        else:
            print("Outside the plot")
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(
            button=0,
            callback=delete_droplet ,
            user_data=app 
            )

