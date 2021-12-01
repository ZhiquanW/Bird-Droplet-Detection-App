import dearpygui.dearpygui as dpg
import os
import numpy as np
import matplotlib.pyplot as plt
tmp_file_path = os.path.join(os.getcwd(), "data/101_reye_3_bf.png")

def add_and_load_image(image_path="", parent=None):
    image_path = tmp_file_path
    width, height, channels, data = dpg.load_image(image_path)
    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)
    with dpg.window(label="Tutorial"):
        dpg.add_image("texture_id")

    # if parent is None:
    #     return dpg.add_image(texture_id)
    # else:
    #     return dpg.add_image(texture_id, parent=parent)


def save_callback():
    print("Save Clicked")


with dpg.window(label="Example Window"):
    dpg.add_text("Hello world")
    dpg.add_button(label="Save", callback=add_and_load_image)
    dpg.add_input_text(label="string")
    dpg.add_slider_float(label="float")
image_path = tmp_file_path
print(image_path)
width, height, channels, data = dpg.load_image(image_path)
data_np = np.frombuffer(data, dtype="int32").reshape(width, height, channels)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
ax.imshow(data_np)
print(data_np[0])
with dpg.texture_registry() as reg_id:
    texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)
with dpg.window(label="Tutorial"):
    dpg.add_image(texture_id)
    data_np[0:10000] = 0


dpg.setup_viewport()
dpg.start_dearpygui()
