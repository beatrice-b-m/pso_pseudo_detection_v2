from tqdm import tqdm
import pandas as pd
import numpy as np
import pydicom


class DataLoader:
    def __init__(self, df_path, seed, df_n: int = -1):
        self.embed_path = '/media/careinfolab/CI_Lab'
        original_df = pd.read_csv(df_path)
        odf_len = len(original_df)
        
        # if df_n is set to -1, loop over the entire dataset
        if df_n == -1:
            df_n = odf_len
            
        # limit dataset length
        self.df = original_df.sample(df_n, random_state=seed)
        self.out_df = self.df.reset_index().copy()
        
    def iterate(self):
        # iterate over dataset
        for i, data in tqdm(self.out_df.iterrows()):
            # get image from dicom
            dcm_path = self.embed_path + data.anon_dicom_path[26:]
            roi_list = self.try_roi(data)
            yield i, Mammogram(dcm_path, roi_list)
            
    def try_roi(self, data):
        roi_coords = data.ROI_coords
        roi_list = self.extract_roi(roi_coords)
               
        return roi_list
    
    def extract_roi(self, roi: str):  # convert ROI string to list
        roi = roi.translate({ord(c): None for c in "][)(,"})
        roi = list(map(int, roi.split()))
        roi_list = []
        for i in range(len(roi) // 4):
            roi_list.append(roi[4*i:4*i+4])
        return roi_list
            
    
        
class Mammogram:
    def __init__(self, dcm_path, roi_list):
#         self.img_dir, self.plots_dir = dirs
        
        self.dcm = pydicom.dcmread(dcm_path)
        self.roi_list = roi_list
        self.img = self.prepare_image(self.dcm.pixel_array)
        self.img_shape = self.img.shape
        
        self.truth_mask = self.get_mask()
        
        self.crop_img = self.segment_tissue(self.img)
        self.crop_shape = self.crop_img.shape
        self.img_dir = ''
        
    def update_img_dir(self, img_dir):
        self.img_dir = img_dir
        
        
    def prepare_image(self, img):
        # duplicate image to get 3 channels
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        
        img_shape = img.shape
        
        # normalize image
        img = self.normalize_image(img)
        
        # correct image laterality
        img = self.correct_laterality(img, img_shape)
        return img
        
    def normalize_image(self, img):
        # normalize image
        norm_img = (img-np.min(img))/(np.max(img)-np.min(img))
        return (norm_img * 255).astype(np.uint8)
        
    def correct_laterality(self, img, img_shape):
        img_side = self.check_side(img)
        if img_side == 'R':
            print('Flipping image')
            img = np.fliplr(img)
            
#             if self.roi_list is not None:
#             print(self.roi_list)
#             self.roi_list = [self.switch_coord_side(roi, img_shape) for roi in self.roi_list]
#             print(self.roi_list)
                
        return img

    def check_side(self, img):
        slice_l = img[:, :100]
        slice_r = img[:, -100:]
        mean_l = slice_l.mean()
        mean_r = slice_r.mean()
        if mean_l > mean_r:
            return 'L'
        elif mean_r > mean_l:
            return 'R'
        else:
            print('Error: Laterality could not be determined!')
            return 'E'
        
    def switch_coord_side(self, roi, img_shape):
        minX = roi[1]
        maxX = roi[3]
        roi[1] = img_shape[1] - maxX
        roi[3] = img_shape[1] - minX
        return roi
        
    def segment_tissue(self, img, pv_thresh: int = 2, pr_thresh: float = 0.99):
        """
        img:        (r, c, 3) uint8 array
        pv_thresh:  max pixel value to consider background (default = 2)
        pr_thresh:  max col/row proportion to consider
                    a background col/row (default = 0.99)
        """
        rows, cols, _ = img.shape

        # find half the height of the image
        half_r = int(rows / 2)

        # count the number of pixels in each row with a value <= pv_thresh
        b_r_bg_c = (img[half_r:, :, 0] <= pv_thresh).sum(axis=1)

        # divide each count by the number of cols to get the bg proportion in each row
        b_r_bg_pr = np.divide(b_r_bg_c, cols)

        # count the number of rows with a bg proportion > pr_thresh
        b_r_bg = rows - (b_r_bg_pr > pr_thresh).sum()

        # count the number of pixels in each col with a value <= pv_thresh
        c_bg_c = (img[:, :, 0] <= pv_thresh).sum(axis=0)

        # divide each count by the number of rows to get the bg proportion in each col
        c_bg_pr = np.divide(c_bg_c, rows)

        # count the number of cols with a bg proportion > pr_thresh
        c_bg = cols - (c_bg_pr > pr_thresh).sum()

        return img[:b_r_bg, :c_bg, :]
    
    def get_mask(self):
        mask = np.zeros((self.img_shape[0], self.img_shape[1]), dtype=np.bool)
        for roi in self.roi_list:
            mask[roi[0]:roi[2], roi[1]:roi[3]] = True
        return mask