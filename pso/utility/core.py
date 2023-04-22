# from utility import tools
from model.classifier import PatchClassifier
from data import data_loader
from opt import particle_swarm as ps
from utility import viz
import matplotlib.pyplot as plt
import random
import numpy as np
import time
from datetime import datetime
import os


# core
class Core:
    def __init__(self, max_vel: float = 50.0, n_clust: int = 8, p_per_clust: int = 8, 
                 model_type: str = 'resnet', gpu_str: str = '0', 
                 patch_width: int = 256, topology: str = 'local'):
        """
        :param max_vel: float
        Maximum particle velocity (default = 50.0)
        :param c_range: tuple[float, float]
        Min and max values for cognitive and social parameters (default = (0.5, 2.5))
        :param n_p:
        Number of particles to initialize (default = 32)
        :param model_type: str
        Default is '152v2' to use ResNet152v2 for inference.
        :param gpu_str: str
        Input GPUs to use during inference (or -1 to force CPU use)
        """

        self.max_vel = max_vel
        self.n_clust = n_clust
        self.p_per_clust = p_per_clust
        self.p_clf = PatchClassifier(gpu_str, model_type)
        self.patch_width = patch_width
        self.topology = topology

    def start(self, df_path, seed: int = 4, n: int = -1, 
              n_it: int = 10, plot_its: bool = False, 
              plot_hm: bool = False, save_plots: bool = False):

        # create individual run folder
        now = datetime.now()
        file_dt = now.strftime("%Y_%m_%d_%H:%M")
        run_dir = f"./logs/run_{file_dt}"
        os.mkdir(run_dir)

        random.seed(seed)
        np.random.seed(seed)
        
        dataset = data_loader.DataLoader(df_path, seed, n)
        dataset.out_df['IOU'] = 0.0
        dataset.out_df['DICE'] = 0.0
        dataset.out_df['time'] = 0.0

        print(f"Total Images: {n}")
        for i, mammo in dataset.iterate():
            print(f'Image {i}')
            start_time = time.process_time()
            
            if save_plots:
                img_dir = f"{run_dir}/img_{i}"
#                 plots_dir = f"{img_dir}/plots"
                os.mkdir(img_dir)
                mammo.update_img_dir(img_dir)
            
            swarm = ps.Swarm(mammo, self.n_clust, self.p_per_clust)
            
            log = swarm.optimize(self.objective, topology=self.topology, 
                                 n_it=n_it, method='max', i=i, 
                                 plot_its=plot_its, save=save_plots)
            
            pred_map = assemble_pred_map(log, mammo.img_shape, self.patch_width)
            
            if plot_hm:
                viz.plot_heatmap(mammo, pred_map, save=save_plots)
            
            threshold = 0.5
            thresh_map = (pred_map > threshold)
            
            iou, dice = evaluate_pred_map(thresh_map, mammo)
            
            print('IOU: ', iou)
            print('DICE: ', dice)
            
            end_time = time.process_time()
            dataset.out_df.loc[i, 'IOU'] = iou
            dataset.out_df.loc[i, 'DICE'] = dice
            dataset.out_df.loc[i, 'time'] = end_time - start_time
            
        out_path = run_dir + '/out_df.csv'
        dataset.out_df.to_csv(out_path, index=False)
        print('Dataset saved to: ', out_path)

    def objective(self, patches):
        # get patch predictions
        predictions = self.p_clf.predict_batch(patches)
        return predictions

    
    
    
class BaseLine:
    def __init__(self, df_path, n, gpu_str: str = '', 
                 model_type: str = 'resnet'):
        self.df_path = df_path
        self.n = n
        
        # load classifier
        self.p_clf = PatchClassifier(gpu_str, model_type)
        
    def start_bl(self, patch_width: int = 256, overlap: float = 0.0, 
                 plot_hm: bool = False, seed: int = 4):
        # load dataset
        dataset = data_loader.DataLoader(self.df_path, seed, self.n)
        
        # make log folder
        now = datetime.now()
        file_dt = now.strftime("%Y_%m_%d_%H:%M")
        run_dir = f"./logs/baseline_{str(overlap)}_{file_dt}"
        os.mkdir(run_dir)

        random.seed(seed)
        np.random.seed(seed)
        
        dataset.out_df['IOU'] = 0.0
        dataset.out_df['DICE'] = 0.0
        dataset.out_df['time'] = 0.0

        print(f"Total Images: {self.n}")
        for img_i, mammo in dataset.iterate():
            start_time = time.process_time()
            print(f'Image {img_i}')
            patch_centers, patch_array = self.tile_image(mammo.crop_img, 
                                                         patch_width, 
                                                         overlap = overlap)
            
            # apply padding to patches for classifier
            pad_patch_array = self.pad_patches(patch_array)
            
            # get patch predictions
            fitness = self.objective(pad_patch_array)
            
            # apply background correction to unpadded patches
            fitness = self.correct_background(patch_array, fitness)
            
            # create log array
            log_array = np.hstack((patch_centers, fitness.reshape(-1, 1)))
            
            pred_map = assemble_pred_map(log_array, mammo.img_shape, patch_width, mode='baseline')
            
            if plot_hm:
                viz.plot_heatmap(mammo, pred_map)
            
            threshold = 0.5
            thresh_map = (pred_map > threshold)
            
            iou, dice = evaluate_pred_map(thresh_map, mammo)
            
            print('IOU: ', iou)
            print('DICE: ', dice)
            
            end_time = time.process_time()
            dataset.out_df.loc[img_i, 'IOU'] = iou
            dataset.out_df.loc[img_i, 'DICE'] = dice
            dataset.out_df.loc[img_i, 'time'] = end_time - start_time
            
        out_path = run_dir + '/out_df.csv'
        dataset.out_df.to_csv(out_path, index=False)
        print('Dataset saved to: ', out_path)
            
    def tile_image(self, array, patch_width, overlap):
        half_width = int(0.5 * patch_width)
        y, x, _ = array.shape

        # create empty arrays
        center_array = np.empty((0, 2))
        patch_array = np.empty((0, patch_width, patch_width, 3), dtype=np.uint8)

        # get overlap width and distance between patches
        overlap_width = int(patch_width * overlap)    
        patch_dist = patch_width - overlap_width

        # get number of steps for patch dist within x-overlap_width
        y_steps = ((y-overlap_width) // patch_dist)
        print(f'{y_steps = }')

        # get number of steps for patch dist within x-overlap_width
        x_steps = ((x-overlap_width) // patch_dist)
        print(f'{x_steps = }')

        # get coord steps for y and x
        y_step_list = [(patch_dist * i) + half_width for i in range(y_steps)]
        x_step_list = [(patch_dist * i) + half_width for i in range(x_steps)]

        for y_center in y_step_list:
            for x_center in x_step_list:
                patch = array[y_center-half_width:y_center+half_width, 
                              x_center-half_width:x_center+half_width, :]
                if patch.shape != (256, 256, 3):
                    continue

                center_array = np.vstack((center_array, np.array([y_center, x_center])))
    #             print(f'{patch.shape = }')
                patch_array = np.vstack((patch_array, patch[np.newaxis, ...]))


        return center_array, patch_array

    def pad_patches(self, patches, out_width = 512):
        pad_array = np.zeros((patches.shape[0], out_width, out_width, 3), dtype=np.uint8)
        offset = int((out_width - patches.shape[1]) / 2)

        pad_array[:, offset:(out_width-offset), offset:(out_width-offset), :] = patches
        return pad_array

    def objective(self, patches):
        # get patch predictions
        predictions = self.p_clf.predict_batch(patches)
        return predictions

    def correct_background(self, patches, fitness, pv_thresh = 5, bg_thresh = 0.7):
        # get mask of pixels lower than the pixel value threshold
        bg_mask = patches < pv_thresh

        # get the proportion of background pixels along axis 0
        bg_percent = np.mean(bg_mask, axis=(1, 2, 3))

        # where the bg_percent is higher than the threshold, replace the fitness with 0
        fitness = np.where(bg_percent > bg_thresh, 0, fitness)
        return fitness
            
            

def assemble_pred_map(pred_log, img_shape, patch_width, mode: str = 'pso'):
    half_width = int(0.5*patch_width)
    
    # get empty arrays to build pred_map with
    pred_map = np.zeros((img_shape[0], img_shape[1]), dtype=float)
    pred_count_map = np.zeros((img_shape[0], img_shape[1]), dtype=int)

    # iterate through preds
    for pred in pred_log:
        y, x, p = pred
        
        if mode != 'baseline':
            p = -p

        y = int(y)
        x = int(x)

        # add prediction to pred_map
        pred_map[y-half_width:y+half_width, x-half_width:x+half_width] += p

        # increment pred_count_map
        pred_count_map[y-half_width:y+half_width, x-half_width:x+half_width] += 1

    # average pred_map
    pred_map = np.divide(pred_map, pred_count_map, where=(pred_count_map > 0))
    return pred_map
    
def evaluate_pred_map(thresh_map, mammo):
    iou = get_iou(thresh_map, mammo.truth_mask)
    dice = get_dice(thresh_map, mammo.truth_mask)
    return iou, dice
    
def get_iou(thresh_map, truth_map):
    # get overlap between maps
    overlap = thresh_map*truth_map

    # get union between maps
    union = thresh_map + truth_map

    # return iou
    return overlap.sum()/float(union.sum())
    
def get_dice(thresh_map, truth_map):
    # compute dice
    return np.sum(thresh_map[truth_map == True]) * 2.0 / (np.sum(thresh_map) + np.sum(truth_map))


if __name__ == "__main__":
    # add unit test
    print('add a unit test here :P')
