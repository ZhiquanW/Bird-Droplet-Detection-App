import dearpygui.dearpygui as dpg
from collections import namedtuple
import core
from matplotlib.pyplot import semilogx, show
import torch
import dpg_utils
import numpy as np
import utils

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])


class cell_info:
    def __init__(self, tex_id, size, loc, image_id=None):
        self.texture_id = tex_id
        self.image_series_id = image_id
        self.size = size
        self.loc = loc
        self.ref = None

    def update_info(self, tex_id, image_id, size, loc):
        self.texture_id = tex_id
        self.image_series_id = image_id
        self.size = size
        self.loc = loc

    def bottom_right(self):
        return [self.loc[0] + self.size[0], self.loc[1] + self.size[1]]


def file_selector_callback(sender, app_data, app):
    img_keys = []
    img_types = []
    img_path = []
    for img_name in app_data["selections"].keys():
        img_path.append(app_data["selections"][img_name])
        name_features = str(img_name).split(".")[0].split("_")
        img_types.append(name_features[-1].lower())
        img_keys.append("_".join(name_features[:-1]))
    if img_keys.count(img_keys[0]) != len(img_keys):
        app.logger.log_warning(
            "two images does not have the same key: {keys}".format(img_keys)
        )
        return
    if not ("bf" in img_types and "e" in img_types):
        app.logger.log_warning(
            "the type of the two images should be 'bf' or 'e': {types}".format(
                types=img_types
            )
        )
        return
    if img_types[0] == "bf":
        img_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0] == "e":
        img_pair = ImgPathPair(bright=img_path[1], blue=img_path[0])

    app.logger.log(
        "record target image paths: \n\t1)[bright] {br_img_name}\n\t2)[blue] {bl_img_name}".format(
            br_img_name=img_pair.bright, bl_img_name=img_pair.blue
        )
    )

    # set new image pair path
    app.img_pair = img_pair
    # update text in main panel
    dpg.set_value(
        "main_panel_bright_img_id",
        value="bright image: {img_name}".format(
            img_name=img_pair.bright.split("/")[-1],
        ),
    )
    dpg.set_value(
        "main_panel_blue_img_id",
        value="blue image: {img_name}".format(
            img_name=img_pair.blue.split("/")[-1],
        ),
    )
    # clear previous images
    dpg_utils.clear_drawlist(app.texture_ids)
    # add images to working space
    if app.xaxis is None:
        app.xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="x axis", parent="image_plot")
    if app.yaxis is None:
        app.yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="y axis", parent="image_plot")
    br_w, br_h = dpg_utils.add_image(img_pair.bright, app.texture_ids[0])
    br_size = np.array([br_w, br_h])
    br_loc = np.array([0, 0])
    br_cell = cell_info(app.texture_ids[0], br_size, br_loc)
    br_tex_ref = dpg.add_image_series(
        app.texture_ids[0],
        br_cell.loc,
        br_cell.bottom_right(),
        label=app.texture_ids[0],
        parent=app.yaxis,
    )
    br_cell.ref = br_tex_ref
    bl_w, bl_h = dpg_utils.add_image(img_pair.blue, app.texture_ids[1])
    bl_size = np.array([bl_w, bl_h])
    bl_loc = np.array([0, 0])
    bl_cell = cell_info(app.texture_ids[1], bl_size, bl_loc)
    bl_tex_ref = dpg.add_image_series(
        app.texture_ids[1],
        bl_cell.loc,
        bl_cell.bottom_right(),
        show=False,
        label=app.texture_ids[1],
        parent=app.yaxis,
    )
    bl_cell.ref = bl_tex_ref
    dpg_utils.add_heatmap_image(br_w, br_h, app.texture_ids[2])
    hm_cell = cell_info(app.texture_ids[2], br_size, br_loc)
    hm_tex_ref = dpg.add_image_series(
        app.texture_ids[2],
        br_cell.loc,
        br_cell.bottom_right(),
        show=False,
        label=app.texture_ids[2],
        parent=app.yaxis,
    )

    hm_cell.ref = hm_tex_ref
    dpg.fit_axis_data(app.xaxis)
    dpg.fit_axis_data(app.yaxis)
    # add cell info to app
    app.gallery.append(br_cell)
    app.gallery.append(bl_cell)
    app.gallery.append(hm_cell)
    # inform app that the image is loaed
    app.image_loaded = True
    enable_all_items(app)
    # print(dpg.get_item_configuration(app.legend))


def check_image_loaded(app):
    if not app.image_loaded:
        app.logger.log_error("images are not loaded")
        return False
    return True


def detect(sender, app_data, d_app):
    if not check_image_loaded(d_app):
        return
    d_app.logger.log("start detection: tpye{d}".format(d=d_app.target_device))

    droplet_num, predicted_map, predicted_heatmap = utils.binary_droplet_detection(
        d_app.img_pair.blue,
        d_app.img_pair.bright,
        d_app.batch_size,
        d_app.padding,
        d_app.stride,
        d_app.winsize,
        threshold=0.7,
        erosion_iter=1,
        model=d_app.models[d_app.target_type],
        device=d_app.target_device,
        verbose=True,
    )
    d_app.logger.log("end detection: {d}".format(d=droplet_num))
    core.app.droplet_locs = utils.droplet_loc(predicted_map)
    print(core.app.droplet_locs)
    utils.pic_rectangle(core.app.droplet_locs)
    # with dpg.window(id="button_window"):
    #     dpg.add_button(label="Add",callback = Add,user_data=app.droplet_locs)
    #     dpg.add_button(label="Delete",callback = Delete,user_data=app.droplet_locs)
    #     pass
    dpg.show_item("button_window")
    return core.app.droplet_locs

    
def Add():
    with dpg.handler_registry():
        print('Add :')
        while 1:
            if dpg.is_item_hovered("image_plot") == True:
                if dpg.is_item_left_clicked("image_plot") ==True:
                    loc = dpg.get_plot_mouse_pos()
                    # locs = droplet_loc(predicted_map)
                    # locs.append([round(loc) for loc in loc])
                    locs = [round(loc) for loc in loc]  
                    core.app.droplet_locs.append(locs)
                    utils.pic_rectangle(core.app.droplet_locs,core.app.size,update=True)
                    break            
    pass

def Delete():
    with dpg.handler_registry():
        print('Delete :')
        while 1:
            if dpg.is_item_hovered("image_plot") == True:
                if dpg.is_item_left_clicked("image_plot") ==True:
                    loc = dpg.get_plot_mouse_pos()
                    locs = [[round(loc) for loc in loc]]
                    # print(locs[0])
                    # [140, 141]
                    try_locs = utils.find_rectangle(locs[0],locs,size=core.app.size)
                    for try_loc in try_locs:
                        if try_loc in core.app.droplet_locs:
                            core.app.droplet_locs.remove(try_loc)
                    print("droplet_locs",core.app.droplet_locs)
                    utils.pic_rectangle(core.app.droplet_locs,core.app.size,update=True)

                    break  

    pass

def Size(sender, app_data, user_data):
    core.app.size = app_data
    return core.app.size


def update_blue_offset(sender, app_data, app):
    if not check_image_loaded(app):
        return
    app.blue_offset[0] = app_data[0]
    app.blue_offset[1] = app_data[1]
    dpg.configure_item(
        app.gallery[1].ref,
        bounds_min=app.gallery[1].loc + app.blue_offset[0],
        bounds_max=app.gallery[1].bottom_right() + app.blue_offset[1],
    )


def switch_texture(sender, app_data, app):
    if not check_image_loaded(app):
        return
    if app_data == "Bright Field":
        dpg.configure_item(app.gallery[0].ref, show=True)
        dpg.configure_item(app.gallery[1].ref, show=False)
        dpg.configure_item(app.gallery[2].ref, show=False)
    elif app_data == "Blue Field":
        dpg.configure_item(app.gallery[0].ref, show=False)
        dpg.configure_item(app.gallery[1].ref, show=True)
        dpg.configure_item(app.gallery[2].ref, show=False)
    elif app_data == "Heatmap":
        dpg.configure_item(app.gallery[0].ref, show=False)
        dpg.configure_item(app.gallery[1].ref, show=False)
        dpg.configure_item(app.gallery[2].ref, show=True)


def update_padding(sender, app_data, app):
    app.padding = app_data


def update_stride(sender, app_data, app):
    app.stride = app_data


def update_win_size(sender, app_data, app):
    app.winsize = app_data


def swtich_target_type(sender, app_data, app):
    names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
    target_type = names.index(app_data)
    app.target_type = target_type
    app.logger.log("ctarget type: {d}".format(d=names[app.target_type]))


def set_device(sender, app_data, app):
    if app_data == "cpu":
        app.target_device = torch.device("cpu")
    elif app_data == "gpu":
        app.logger.log("cuda available: {d}".format(d=torch.cuda.is_available()))
        if torch.cuda.is_available():
            app.target_device = torch.device("cuda")
    app.logger.log("current device: {d}".format(d=app.target_device))


def enable_all_items(app):
    for key, val in app.item_dict.items():
        dpg.enable_item(val)
