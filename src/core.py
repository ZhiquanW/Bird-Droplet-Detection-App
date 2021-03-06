from faulthandler import disable
from matplotlib.pyplot import legend, show
import numpy as np
import dearpygui.dearpygui as dpg
from numpy.lib.function_base import append
from torch.nn.modules.module import T
import callbacks
import torch
import os
import torch.nn as nn
from PIL import ImageOps
from tags import *


class bio_image_vgg_classification_net(nn.Module):
    def __init__(self, class_num: int = 6, dropout_ratio: float = 0.1):
        super(bio_image_vgg_classification_net, self).__init__()
        self.class_num = 6  # 5 types of bird droplet + 1 background type
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=96, kernel_size=7, stride=2, padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(6 * 6 * 512, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, class_num, bias=True)
        self.dropout_layer = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        layer0_x = self.pool1(nn.functional.relu(self.conv1(x)))
        layer1_x = self.pool2(nn.functional.relu(self.conv2(layer0_x)))
        layer2_x = nn.functional.relu(self.conv3(layer1_x))
        layer3_x = nn.functional.relu(self.conv4(layer2_x))
        layer4_x = self.pool1(nn.functional.relu(self.conv4(layer3_x)))

        layer4_x = layer4_x.view(-1, 6 * 6 * 512)
        layer5_x = nn.functional.relu(self.dropout_layer(self.fc1(layer4_x)))
        layer6_x = nn.functional.relu(self.dropout_layer(self.fc2(layer5_x)))
        layer7_x = self.dropout_layer(self.fc3(layer6_x))
        pred = torch.sigmoid(layer7_x)
        return pred

    @torch.no_grad()
    def get_all_preds(model, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, _ = batch
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)
        return all_preds


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, *, images, padding: int, win_size, stride, transform):
        e_image, bf_image = images
        img_h, img_w = e_image.size
        self.padded_e_image = ImageOps.expand(
            e_image, (padding, padding, padding, padding)
        )
        self.padded_bf_image = ImageOps.expand(
            bf_image, (padding, padding, padding, padding)
        )
        self.MAP_H, self.MAP_W = (
            (np.array([img_h, img_w]) + 2 * padding - win_size) / stride + 1
        ).astype(int)

        self.transform = transform
        self.padding = padding
        self.win_size = win_size
        self.stride = stride
        self.cell_image_num = self.MAP_W * self.MAP_H

    def __len__(self):
        return self.cell_image_num

    def __getitem__(self, idx):
        h = int(idx / self.MAP_W)
        w = idx % self.MAP_W
        top = h * self.stride
        bottom = top + self.win_size
        left = w * self.stride
        right = left + self.win_size
        e_slided = self.padded_e_image.crop((left, top, right, bottom))
        bf_slided = self.padded_bf_image.crop((left, top, right, bottom))
        e_transformed = self.transform(e_slided)
        bf_transformed = self.transform(bf_slided)
        image_mat = torch.cat((e_transformed, bf_transformed), 0)
        return image_mat, 0


class app:
    def __init__(self) -> None:
        self.img_pair = callbacks.ImgPathPair(bright=None, blue=None)
        self.models = []
        self.detection_data = [[], [], [], [], []]
        # store raw images (Bright, Blue, Heatmap)
        self.img_dtexture_list = [None,None,None]
        # store detected rectangles (0,1,2,3,4,5)
        self.detection_notation_list = []
        # store target area rectangel
        self.target_area_dtexture = None
        self.target_area_top_left = [0, 0]
        self.target_area_bottom_right = [0, 0]
        self.image_spacing = 20
        self.xaxis = None
        self.yaxis = None
        self.blue_offset = np.array([0, 0])
        self.legend = None
        self.batch_size = 64
        self.winsize = 10
        self.padding = 7
        self.stride = 2
        self.target_type = 0
        self.target_device = torch.device("cpu")
        self.image_loaded = False
        self.default_font = None
        self.item_tag_dict = {}
        self.rect_item_tag_dict = {}
        self.rectangle_size = 15
        self.target_type_names = (
            "Type One",
            "Type Two",
            "Type Three",
            "Type Four",
            "Type Five",
        )
        self.droplet_dict_locs = {
            "Type One": [],
            "Type Two": [],
            "Type Three": [],
            "Type Four": [],
            "Type Five": [],
        }
        self.droplet_dict_num = {
            "Type One": [],
            "Type Two": [],
            "Type Three": [],
            "Type Four": [],
            "Type Five": [],
        }
        self.droplet_dict_colors = {
            "Type One": (255, 0, 0, 255),
            "Type Two": "white",
            "Type Three": "green",
            "Type Four": "yellow",
            "Type Five": "blue",
        }
        # UI params
        self.enable_manual_detection_mode = False
        self.display_raw_texture_type = 0

        # export params
        self.density_file_path = "./default_densities.csv"
        self.distance_file_path = "./default_distance.csv"

    def __load_models(self):
        for i in range(5):
            model = torch.load(
                os.path.join(os.getcwd(), "models/mt{t}".format(t=i)),
                map_location=torch.device("cpu"),
            )
            model.eval()
            self.models.append(model)
        print("model loaded")

    def _create_file_selectors(self):
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            id=item_tags.file_dialog_image_select,
            file_count=2,
            callback=callbacks.image_selector_callback,
            user_data=self,
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".tiff", color=(0, 255, 255, 255))
            dpg.add_file_extension(".tif", color=(0, 255, 255, 255))
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            id=item_tags.export_data_file_selector,
            file_count=1,
            callback=callbacks.set_density_data_file,
            user_data=self,
        ):
            # dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".csv", color=(255, 0, 0, 255))

    def __set_font(self):
        # add a font registry
        with dpg.font_registry():
            # add font (set as default for entire app)
            self.default_font = dpg.add_font("Retron2000.ttf", 15)
        dpg.bind_font(self.default_font)
        print("font set")

    def __create_ui_layout(self):
        self._create_file_selectors()
        self.__create_main_panel()

    def __create_main_panel(self):
        with dpg.window(label="Main", tag=item_tags.main_window):
            dpg.add_plot(
                    label="Image Plot",
                    height=-1,
                    width=-1,
                    tag=item_tags.image_plot_workspace,
                    equal_aspects=True,
                    crosshairs=True,
                    box_select_button=True,
                )
            dpg.add_plot_legend(parent=item_tags.image_plot_workspace)

            self.xaxis = dpg.add_plot_axis(
                dpg.mvXAxis,
                label="x axis",
                invert=False,
                parent=item_tags.image_plot_workspace,
            )
            self.yaxis = dpg.add_plot_axis(
                dpg.mvYAxis,
                label="y axis",
                invert=True,
                parent=item_tags.image_plot_workspace,
            )
               
                # with dpg.window(width=500, height=300):
                #     dpg.add_text("Click me with any mouse button", tag="text item")
                #     dpg.add_text("Close window with arrow to change visible state printing to console", tag="text item 2")

            with dpg.window():
                with dpg.collapsing_header(
                    label="Image Information", default_open=True
                ):
                    self.item_tag_dict[item_tags.image_selector] = dpg.add_button(
                        label="Image Selector",
                        callback=lambda: dpg.show_item(
                            item_tags.file_dialog_image_select
                        ),
                    )
                    dpg.add_text(
                        "bright image: {img_name}".format(
                            img_name=""
                            if self.img_pair.bright is None
                            else self.img_pair.bright.split("/")[-1],
                        ),
                        tag="main_panel_bright_img_id",
                    )
                    dpg.add_text(
                        "blue image: {img_name}".format(
                            img_name=""
                            if self.img_pair.blue is None
                            else self.img_pair.blue.split("/")[-1],
                        ),
                        tag="main_panel_blue_img_id",
                    )
                with dpg.collapsing_header(
                    label="Workspace Operation", default_open=True
                ):
                    self.item_tag_dict[
                        item_tags.blue_img_offset_slider
                    ] = dpg.add_slider_intx(
                        label="blue image offset",
                        size=2,
                        callback=callbacks.update_blue_offset,
                        user_data=self,
                        enabled=True,
                        min_value=-100,
                        max_value=100,
                        width=200,
                    )
                    self.item_tag_dict[
                        item_tags.display_texture_radio
                    ] = dpg.add_radio_button(
                        ("Bright_Field", "Blue_Field", "Heatmap"),
                        horizontal=True,
                        user_data=self,
                        callback=callbacks.select_display_raw_texture,
                        enabled=True,
                    )
                    self.item_tag_dict[
                        item_tags.maunal_mode_radio
                    ] = dpg.add_checkbox(
                        label="maunal mode",
                        user_data=self,
                        callback=callbacks.switch_droplet_manual_detectio_mode,
                        default_value=self.enable_manual_detection_mode,
                    )
                    self.rect_item_tag_dict["rect_size"] = dpg.add_slider_int(
                        label="rectangle size",
                        default_value=self.rectangle_size,
                        min_value=5,
                        max_value=30,
                        callback=callbacks.set_rect_size,
                        user_data=self,
                    )
                    self.rect_item_tag_dict["rect_color"] = dpg.add_color_picker(
                        label="rectangle color",
                        display_hex=False,
                        callback=callbacks.rect_color,
                        user_data=self,
                    )
                with dpg.collapsing_header(
                    label="Detection Configuration", default_open=True
                ):
                    self.item_tag_dict[
                        item_tags.target_area_top_left_slider
                    ] = dpg.add_slider_intx(
                        label="target area: top left",
                        size=2,
                        callback=callbacks.update_target_area_top_left,
                        user_data=self,
                        enabled=True,
                        min_value=0,
                        max_value=1500,
                    )
                    self.item_tag_dict[
                        item_tags.target_area_bottom_right_slider
                    ] = dpg.add_slider_intx(
                        label="target area: bottom right",
                        size=2,
                        callback=callbacks.update_target_area_bottom_right,
                        user_data=self,
                        enabled=True,
                        min_value=0,
                        max_value=1500,
                    )
                    self.item_tag_dict[
                        item_tags.device_selector
                    ] = dpg.add_radio_button(
                        ("cpu", "gpu"),
                        default_value="cpu",
                        horizontal=True,
                        callback=callbacks.set_device,
                        user_data=app,
                        enabled=True,
                    )

                    self.item_tag_dict[
                        item_tags.target_droplet_type_combo
                    ] = dpg.add_combo(
                        self.target_type_names,
                        label="target type",
                        default_value=self.target_type_names[0],
                        user_data=self,
                        callback=callbacks.swtich_target_type,
                    )

                    self.item_tag_dict["padding"] = dpg.add_input_int(
                        label="Padding", default_value=7, enabled=True
                    )
                    self.item_tag_dict["stride"] = dpg.add_input_int(
                        label="Stride", default_value=2, enabled=True
                    )
                    self.item_tag_dict["winsize"] = dpg.add_input_int(
                        label="Window Size", default_value=10, enabled=True
                    )
                    self.item_tag_dict[
                        item_tags.crop_target_area_bottom
                    ] = dpg.add_button(
                        label="crop target area",
                        callback=callbacks.crop_target_area,
                        user_data=self,
                        enabled  = False,
                        show=False
                    )

                    self.item_tag_dict[
                        item_tags.auto_detection_button
                    ] = dpg.add_button(
                        label="auto detection",
                        callback=callbacks.detect_droplets,
                        user_data=self,
                    )
                with dpg.collapsing_header(label="Data Export", default_open=True):
                    self.item_tag_dict[
                        item_tags.export_data_file_selector
                    ] = dpg.add_button(
                        label="Output File Selector",
                        callback=lambda: dpg.show_item(
                            item_tags.export_data_file_selector
                        ),
                    )
                    self.item_tag_dict[item_tags.export_path_txt] = dpg.add_text(
                        dpg.get_value(value_tags.export_data_file_path),
                        tag=item_tags.export_path_txt,
                    )
                    self.item_tag_dict[
                        item_tags.save_image_button
                    ] = dpg.add_button(
                        label="save location",
                        callback=callbacks.export_location,
                        user_data=self,
                    )
                    self.item_tag_dict[item_tags.save_density_data_button] = dpg.add_button(
                        label="save density data",
                        callback=callbacks.export_density_data,
                        user_data=self,
                    )
                    self.item_tag_dict[item_tags.save_distance_data_button] = dpg.add_button(
                        label="save distance data",
                        callback=callbacks.export_distances_data,
                        user_data=self,
                    )

    def handler_registry(self):
        # with dpg.item_handler_registry(tag="widget handler") as handler:
        #     dpg.add_item_visible_handler(callback=callbacks.debug_callbacks)
        # dpg.bind_item_handler_registry("text item", "widget handler")

        with dpg.handler_registry(tag=item_tags.workspace_handler) as handler:
            dpg.add_mouse_click_handler(
                button=0, callback=callbacks.operate_droplet_manually, user_data=self
            )
        # dpg.bind_item_handler_registry(
        #    "text", item_tags.workspace_handler
        # )
        with dpg.handler_registry(tag=item_tags.keyboard_handler):
            dpg.add_key_release_handler(
                key=dpg.mvKey_Q,
                callback=callbacks.switch_display_raw_texture,
                user_data=self,
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_M,
                callback=callbacks.switch_droplet_manual_detectio_mode,
                user_data=self,
            )
        # dpg.bind_item_handler_registry(
        #     item_tags.image_plot_workspace, item_tags.keyboard_handler
        # )

    def launch(self):
        dpg.create_context()
        dpg.create_viewport(title="Oil Droplet Detection", width=3840, height=2160)
        self.__load_models()
        self.__set_font()
        self.__create_ui_layout()
        self.handler_registry()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(item_tags.main_window, True)
        self.debug_load_images()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def debug_load_images(self):
        app_data = {
            "file_path_name": "/home/zhiquan/projects/Bird-Droplet-Detection-App/src/data/2 files Selected.*",
            "file_name": "2 files Selected.*",
            "current_path": "/home/zhiquan/projects/Bird-Droplet-Detection-App/src/data",
            "current_filter": ".*",
            "selections": {
                "PKDK_105_LE_PB_7-6-2020_Acquired Images_09-24-2020_030_BF.tif":os.path.join(os.getcwd(),"data\\mall107_re_pb_10-18-2021_670_BF.tif"),
                "PKDK_105_LE_PB_7-6-2020_Acquired Images_09-24-2020_030_E.tif": os.path.join(os.getcwd(),"data\\mall107_re_pb_10-18-2021_670_E.tif")
            },
        }
        callbacks.image_selector_callback(None, app_data, self)


if __name__ == "__main__":
    app = app()
    app.launch()
