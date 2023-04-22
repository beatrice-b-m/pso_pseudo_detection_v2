import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# code for result visualizations
def plot_swarm(log_array, mammo, p_radius, j: int, display: bool = True, save: bool = False):  # CAT = viz

    fig, ax = plt.subplots()
    ax.imshow(mammo.crop_img)
    for i in range(log_array.shape[0]):
        y_cent, x_cent, pred = log_array[i, :]
        
        rec_patch = pat.Rectangle((x_cent - p_radius,
                                   y_cent - p_radius),
                                  2*p_radius,
                                  2*p_radius,
                                  edgecolor=None,
                                  fc=plt.cm.autumn(-pred),
                                  alpha=0.5)
        ax.add_patch(rec_patch)

    # plot patch and rad coords if they exist
    plot_rois(mammo.roi_list, ax)

    # if save is True, save the plot
    if save:
        # when save is true:
        # if the unique image folder does not exist, create it
        ax.set_title(f'Iteration {j}')
        plt.savefig(f'{mammo.img_dir}/iter_{j}_swarm.png', dpi=300, bbox_inches='tight')

    # if display is True, display the plot
    if display:
        plt.show()


# MOVE BELOW THIS TO VIZ ----------------------------------------------
def plot_heatmap(mammo, float_map, display: bool = True, save: bool = False):
    print('Processing log map...')
    uint8_map = (float_map * 255).astype(np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(mammo.img)

    ax.imshow(uint8_map, cmap='inferno', alpha=float_map, interpolation='bilinear')

    # plot patch and rad coords if they exist
    plot_rois(mammo.roi_list, ax)

    # if save is True, save the plot
    if save:
        # when save is true:
        # if the unique image folder does not exist, create it
        plt.savefig(f'{mammo.img_dir}/final_heatmap.png', dpi=300, bbox_inches='tight')

    # if display is True, display the plot
    if display:
        plt.show()


def plot_rois(roi_list, ax):
    # plot the radiologist drawn ROI
    for r_c in roi_list:
        rec_rad = pat.Rectangle((max(r_c[1], r_c[3]),
                                 max(r_c[0], r_c[2])),
                                -(max(r_c[1], r_c[3]) - min(r_c[1], r_c[3])),
                                -(max(r_c[0], r_c[2]) - min(r_c[0], r_c[2])),
                                edgecolor='xkcd:bright green', fc='None')

        ax.add_patch(rec_rad)

