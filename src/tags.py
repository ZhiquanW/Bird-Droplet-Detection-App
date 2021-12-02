from dearpygui.dearpygui import file_dialog

import dearpygui.dearpygui as dpg
import uuid

from numpy import iinfo


class item_tags:
    file_dialog_image_select = "file_dialog_" + str(uuid.uuid4())
    image_plot_workspace = "image_plot_workspace" + str(uuid.uuid4())
    main_window = "main_window" + str(uuid.uuid4())
    workspace_handler = "workspace_handler" + str(uuid.uuid4())
    keyboard_handler = "keyboard_handler" + str(uuid.uuid4())
    texture_tags = ["Bright_Field", "Blue_Field", "Heatmap"]
    detection_tags = ["type_one", "type_two", "type_three", "type_four", "type_five"]
    maunal_mode_radio = "maunal_mode_radio" + str(uuid.uuid4())
    auto_detection_button = "auto_detection_button"+str(uuid.uuid4())
    target_droplet_type_combo = "target_droplet_type_combo" + str(uuid.uuid4())
    display_texture_radio = "display_texture_radio" + str(uuid.uuid4())
    image_selector = "image_selector"+str(uuid.uuid4())
    device_selector = "device_selector" + str(uuid.uuid4())
    target_area_slider = "target_area_slider" + str(uuid.uuid4())
    blue_img_offset_slider = "blue_img_offset_slider" + str(uuid.uuid4())
    crop_target_area_bottom = "crop_target_area_bottom" + str(uuid.uuid4())