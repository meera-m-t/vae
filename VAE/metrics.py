import numpy as np
def change_range(set_, new_min, new_max):
    set_ = set_.reshape(set_.shape[0]*set_.shape[1])
    OldRange = (max(set_) - min(set_))  
    NewRange = (new_max - new_min) 
    new_value = lambda i: (((i - min(set_)) * NewRange) / OldRange) + new_min
    new_value = np.vectorize(new_value)
    return new_value(set_).reshape(set_.shape[0], set_.shape[1])