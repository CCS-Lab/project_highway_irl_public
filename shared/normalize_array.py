import numpy as np

def normalize_array(array):
    array = np.array(array)
    min_array = np.min(array)
    max_array = np.max(array)
    
    normalized_array = (array-min_array)/(max_array-min_array)
    
    return normalized_array