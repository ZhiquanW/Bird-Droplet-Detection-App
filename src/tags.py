from dearpygui.dearpygui import file_dialog

import dearpygui.dearpygui as dpg
import uuid

from numpy import iinfo


class item_tags:
    file_dialog_image_select = "file_dialog_" + str(uuid.uuid4())
    export_data_file_selector = "export_data_file_select" + str(uuid.uuid4())
    image_plot_workspace = "image_plot_workspace" + str(uuid.uuid4())
    main_window = "main_window" + str(uuid.uuid4())
    workspace_handler = "workspace_handler" + str(uuid.uuid4())
    keyboard_handler = "keyboard_handler" + str(uuid.uuid4())
    texture_tags = ["Bright_Field", "Blue_Field", "Heatmap"]
    detection_tags = ["type_one", "type_two", "type_three", "type_four", "type_five"]
    target_ara_texture = "target_area_texture" + str(uuid.uuid4())
    maunal_mode_radio = "maunal_mode_radio" + str(uuid.uuid4())
    auto_detection_button = "auto_detection_button"+str(uuid.uuid4())
    target_droplet_type_combo = "target_droplet_type_combo" + str(uuid.uuid4())
    display_texture_radio = "display_texture_radio" + str(uuid.uuid4())
    image_selector = "image_selector"+str(uuid.uuid4())
    device_selector = "device_selector" + str(uuid.uuid4())
    target_area_top_left_slider = "target_area_top_left_slider" + str(uuid.uuid4())
    target_area_bottom_right_slider = "target_area_bottom_right_slider" + str(uuid.uuid4())
    blue_img_offset_slider = "blue_img_offset_slider" + str(uuid.uuid4())
    crop_target_area_bottom = "crop_target_area_bottom" + str(uuid.uuid4())
    save_image_button = "save_image_button" + str(uuid.uuid4())
    export_path_txt = "export_path_txt" + str(uuid.uuid4())
class value_tags:
    export_data_file_path = "export_data_file_path"+str(uuid.uuid4())