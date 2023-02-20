import ujson
import torch
import pickle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
import importlib
import utils.fflow as flw
import numpy as np
import io
import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    task_name = './fedtasksave/mnist_cnum25_dist0_skew0_seed0'
    exp_name = 'R51_P0.30_alpha0.07'
    with open(os.path.join(task_name, exp_name, 'record', 'history51.pkl'), 'rb') as test_f:
        hist_unlearn = CPU_Unpickler(test_f).load()
    print("Main accuracy: {}".format(hist_unlearn['accuracy_unlearn'][0]))
    print("Backdoor accuracy: {}".format(hist_unlearn['accuracy_unlearn'][1]))
    # pass