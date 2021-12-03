from typing import List
import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from PIL import Image, ImageOps, ImageDraw

class drawable_texture:
    def __init__(self, texture_tag, image_series_tag,data, size, top_left):
        self.texture_tag = texture_tag
        self.image_series_tag = image_series_tag
        self.data = data
        self.size = size
        self.top_left = top_left

    def update_info(self, texture_tag, image_series_tag, size, top_left):
        self.texture_tag = texture_tag
        self.image_series_tag = image_series_tag
        self.size = size
        self.top_left = top_left

    @property
    def bottom_right(self):
        return self.top_left + self.size


def register_texture(imgae_path, tag):
    im = plt.imread(imgae_path) / 255.0
    height, width, channels = im.shape
    im = np.append(im, np.ones((height, width, 1)), 2)
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width, height, im, tag=tag)
    return width, height ,im

def add_img_texture_to_workspace(image_path, texture_tag, parent_axis, show=False):
    img_w, img_h ,im = register_texture(image_path, texture_tag)
    img_size = np.array([img_w, img_h],dtype=np.int)
    img_top_left = np.array([0, 0],dtype=np.int)
    img_bottom_right = img_top_left + img_size
    img_series_tag = dpg.add_image_series(
        texture_tag,
        img_top_left,
        img_bottom_right,
        show=show,
        label=texture_tag,
        parent=parent_axis,
    )
    img_cell = drawable_texture(texture_tag, img_series_tag,im, img_size, img_top_left)
    return img_cell


def add_notation_buff_to_workspace(img_size, img_buff_tag, parent_axis, show=False,transparent= False):
    dtext_buffer = register_image_buffer(img_size[0], img_size[1], img_buff_tag, transparent)
    img_top_left = np.array([0, 0],dtype=np.int)
    img_bottom_right = img_top_left + img_size
    img_series_tag = dpg.add_image_series(
        img_buff_tag,
        img_top_left,
        img_bottom_right,
        show=show,
        label=img_buff_tag,
        parent=parent_axis,
    )

    dtexture = drawable_texture(img_buff_tag, img_series_tag,dtext_buffer, img_size, img_top_left)
    return dtexture


def register_image_buffer(w, h, tag,transparent):
    w = int(w)
    h = int(h)
    texture_buffer = np.ones((h, w, 4))
    if transparent:
        texture_buffer[:,:,-1] = 0.0
    with dpg.texture_registry():
        dpg.add_dynamic_texture(w, h, texture_buffer.flatten(), tag=tag)
    return texture_buffer


def update_detection_result(app):
    
    pass


def clear_drawlist(img_ids):
    for img_id in img_ids:
        if dpg.does_item_exist(img_id):
            dpg.delete_item(img_id)
            dpg.remove_alias(img_id)
       
def clear_dtextures(dtextures:List[drawable_texture]):
    for dtext in dtextures:
        if dpg.does_item_exist(dtext.image_series_tag):
            dpg.delete_item(dtext.image_series_tag)
            # dpg.remove_alias(dtext.image_series_tag)
        if dpg.does_item_exist(dtext.texture_tag):
        #     dpg.delete_item(dtext.texture_tag)
            dpg.remove_alias(dtext.texture_tag)

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])

def parse_image_selector_data(app_data):
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
        img_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0] == "e":
        img_pair = ImgPathPair(bright=img_path[1], blue=img_path[0])

def set_heatmap(predicted_heatmap):
    dpg.set_value("Heatmap" , predicted_heatmap)


def draw_target_area_dtexture(dtexture:drawable_texture,h0,w0,h1,w1):
    im = Image.fromarray(np.uint8(dtexture.data),mode='RGBA')
    im.putalpha(0)
    im_draw = ImageDraw.Draw(im)
    im_draw.rectangle((w0,h0,w1,h1), outline="white", width=1)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    img_arr = np.array(im)
    dpg.set_value(dtexture.texture_tag, img_arr.flatten())
