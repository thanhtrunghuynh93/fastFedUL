from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from torch.autograd import Variable
from yaml import load
import numpy as np
import os.path as osp
import torch.utils.data as data
import os
import ujson
import pickle
import torch
import torch.nn.functional as F

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='cifar10',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/cifar10/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: F.pad(
                                            Variable(x.unsqueeze(0), requires_grad=False),
                                            (4,4,4,4),mode='reflect').data.squeeze()),
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                        ])
        self.train_data = datasets.CIFAR10(self.rawdata_path, train=True, download=True, transform=transform_train)
        self.test_data = datasets.CIFAR10(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        # torch.manual_seed(0)
        # self.train_data, self.test_data = data.random_split(datasets.CIFAR10(self.rawdata_path, train=True, download=True, transform=transform_train), [40000, 10000])
        
    # def load_data(self):
    #     data_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     ])
    #     torch.manual_seed(0) # 0, 12, 13
    #     self.train_data, self.attacked_data = data.random_split(datasets.CIFAR10(self.rawdata_path, train=True, download=True, transform=data_transform),
    #                                                             [40000, 10000])
    #     self.test_data = datasets.CIFAR10(self.rawdata_path, train=False, download=True, transform=data_transform)
    #     print(len(list(self.test_data)))
    
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

    def load_southwest_data(self, root):
        with open(osp.join(root, 'southwest_images_new_train.pkl'), 'rb') as train_f:
            x_train = pickle.load(train_f)
        with open(osp.join(root, 'southwest_images_new_test.pkl'), 'rb') as test_f:
            x_test = pickle.load(test_f)
        # x_train = np.moveaxis(x_train, -1, 1).astype('float32')
        # x_test = np.moveaxis(x_test, -1, 1).astype('float32')
        self.ardis_x = np.concatenate((x_train, x_test)).astype('float32')
        self.ardis_y = np.array([0 for i in range(self.ardis_x.shape[0])]).astype('float32')
        self.ardis_x = self.transform_southwest(self.ardis_x)
        print(self.ardis_x[0])
    # def load_attack_cifar10(self):
    #     x_train = [self.attacked_data[did][0].tolist() for did in range(len(self.attacked_data))]
    #     y_train = [self.attacked_data[did][1] for did in range(len(self.attacked_data))]
    #     ## reshape image
    #     self.ardis_x = np.array(x_train).astype('float32')
    #     self.ardis_y = np.array(y_train)
    #     self.ardis_x = self.transform_southwest(self.ardis_x)

    def transform_southwest(self, dataset):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: F.pad(
                                                            Variable(x.unsqueeze(0), requires_grad=False),
                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # AddGaussianNoise(0., 0.05),
                                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])])
        southwest_dset = []
        # dataset = np.moveaxis(dataset, 1, -1) ## add this
        for idx_img in range(dataset.shape[0]):
            southwest_dset.append(transform(dataset[idx_img]).cpu().detach().numpy())
        return np.array(southwest_dset)
    
    def get_dirty_data(self, root = 'customdata/southwest_airline', true_label=0, dirty_label=9, attack_client_idx=0, attack_ratio=0.2):
        attack_client_idx = min(attack_client_idx, self.num_clients-1)
        print('Saving attack data ...')
        self.load_southwest_data(root)
        # self.load_attack_cifar10()
        
        # get index of src num
        dirty_idx = np.nonzero(self.ardis_y == true_label)[0].tolist()
        print('length of clean data in attcked-client:')
        print(len(self.train_cidxs[attack_client_idx]))
        # for idxx in range(10):
        #     print(len(self.train_cidxs[idxx]))
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
