import matplotlib.pyplot as plt
import pytiff
import numpy as np
if __name__ == "__main__":
    print('start')
    img= pytiff.Tiff("../data/mall107_re_pb_10-18-2021_670_E.tif")
    print(np.array(img)/255.0)
    # read an image in the current TIFF directory as a numpy array
    imgplot = plt.imshow(img)
    plt.show()
    print("done")
