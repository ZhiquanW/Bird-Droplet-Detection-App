
import numpy as np
def droplet_locs(predicted_map,top_left):
    from matplotlib import pyplot as plt
    plt.imshow(predicted_map, interpolation='nearest')
    plt.show()
    cell_bool = predicted_map > 0
    locs = np.where(cell_bool)
    x_locs = locs[0] 
    y_locs = locs[1] 
    list_x_locs = x_locs.tolist()
    list_y_locs = y_locs.tolist()
    locs = []
    for i in range(0, len(list_x_locs)):
        locs.append([list_y_locs[i]+top_left[0], list_x_locs[i]+top_left[1]])
    return locs

if __name__ == "__main__":
    tmp_map = np.zeros((20,20))
    tmp_map[4,4] = 1
    locs = droplet_locs(tmp_map,[2,3])
    print(locs)