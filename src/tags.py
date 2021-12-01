from dearpygui.dearpygui import file_dialog

import dearpygui.dearpygui as dpg
import uuid


class item_tags:
    file_dialog_image_select = "file_dialog_" + str(uuid.uuid4())
    image_plot_workspace = "image_plot_workspace" + str(uuid.uuid4())
    main_window = "main_window" + str(uuid.uuid4())
    workspace_handler = "workspace_handler" + str(uuid.uuid4())
    keyboard_handler = "keyboard_handler" + str(uuid.uuid4())
    texture_tags = ["Bright_Field", "Blue_Field", "Heatmap"]
    detection_tags = ["type_one", "type_two", "type_three", "type_four", "type_five"]
    maunal_mode_radio = "maunal_mode_radio" + str(uuid.uuid4())

