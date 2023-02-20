from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader

from yaml import load
import numpy as np
import os.path as osp
import os
import ujson

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='mnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/mnist/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data = datasets.MNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.MNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class CustomTaskGen(TaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5):
        super().__init__(dist_id, num_clients, skewness)

    def load_ARDIS_data(self, root):
        x_train=np.loadtxt(osp.join(root, 'ARDIS_train_2828.csv'), dtype='float')
        x_test=np.loadtxt(osp.join(root, 'ARDIS_test_2828.csv'), dtype='float')
        y_train=np.loadtxt(osp.join(root, 'ARDIS_train_labels.csv'), dtype='float')
        y_test=np.loadtxt(osp.join(root, 'ARDIS_test_labels.csv'), dtype='float')
        ## reshape image
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
        ##
        y_train = np.nonzero(y_train)[1]
        y_test = np.nonzero(y_test)[1]
        self.ardis_x = np.concatenate((x_train, x_test))
        self.ardis_y = np.concatenate((y_train, y_test))

    def get_dirty_data(self, root = 'customdata/ARDIS_DATASET_IV', true_label=7, dirty_label=1, attack_client_idx=0, attack_ratio=0.2):
        attack_client_idx = min(attack_client_idx, self.num_clients-1)
        print('Saving attack data ...')
        self.load_ARDIS_data(root)
        # get index of src num
        dirty_idx = np.nonzero(self.ardis_y == true_label)[0].tolist()
        print('length of clean data in attcked-client:')
        print(len(self.train_cidxs[attack_client_idx]))
        
        attack_len = int(len(self.train_cidxs[attack_client_idx])*attack_ratio)
        print('length of backdoor:')
        print(len(dirty_idx)-attack_len)
        
        attack_len = min(len(dirty_idx), max(1, attack_len))
        test_dirty_idx = dirty_idx[attack_len:]
        dirty_idx = dirty_idx[:attack_len]
        # convert dirty data to save
        y = [dirty_label for i in range(attack_len)]
        x = [self.ardis_x[i].tolist() for i in dirty_idx]
        
        # create a backdoor test
        backdoor_x = [self.ardis_x[i].tolist() for i in test_dirty_idx]
        backdoor_y = [dirty_label for i in range(len(test_dirty_idx))]
        self.test_backdoor = {'x': backdoor_x, 'y': backdoor_y}
        
        # to json file
        self.to_json(attack_client_idx, x, y)
        print('Done.')

    def to_json(self, attack_client_idx, x, y):
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data,
            'dbackdoor': self.test_backdoor
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain':{
                    'x':[self.train_data['x'][did] for did in self.train_cidxs[cid]], 'y':[self.train_data['y'][did] for did in self.train_cidxs[cid]]
                },
                'dvalid':{
                    'x':[self.train_data['x'][did] for did in self.valid_cidxs[cid]], 'y':[self.train_data['y'][did] for did in self.valid_cidxs[cid]]
                }
            }
        feddata[self.cnames[attack_client_idx]]['dtrain']['x'] += x
        feddata[self.cnames[attack_client_idx]]['dtrain']['y'] += y
        with open(os.path.join(self.taskpath, 'attack.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

