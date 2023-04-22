import os
import numpy as np
from numba import njit
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input


# rn152 core
class PatchClassifier:
    def __init__(self, gpu_str, mod_type):
        self.mod_type = mod_type
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        if mod_type == 'resnet':
            # load model
            mod_dir: str = './model/birads0model_resnet152v2_pt_c'
            self.saved_model = load_model(mod_dir, compile=False)
            print('Model loaded')
        elif mod_type == 'test':
            self.saved_model = None
            print('Model loaded')
        else:
            print(f"Error: Model type '{mod_type}' is not recognized!")
            self.saved_model = None

    def predict_batch(self, array):  # CAT = model
        if self.mod_type == 'resnet':
            predictions = self.saved_model.predict(preprocess_input(array))
            predictions = np.squeeze(predictions)
            
        elif self.mod_type == 'test':
            # normalize patches
            arr_min = np.min(array)
            arr_max = np.max(array)
            norm_array = (array - arr_min) / (arr_max - arr_min)
            
            # get average pixel value to test pso
            predictions = np.average(norm_array, axis=(1, 2, 3))
        else:
            print(f"Error: Model type '{self.mod_type}' is not recognized!")
        return predictions
