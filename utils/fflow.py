import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import copy
sample_list=['uniform', 'md', 'active']
agg_list=['uniform', 'weighted_scale', 'weighted_com', 'none']
optimizer_list=['SGD', 'Adam']

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')

    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='none')
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    
    ## For Bach
    ## hyper-parameters of training
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2) 
    parser.add_argument('--theta_delta', help='coefficient multiply with delta each round', type=float, default=1) 
    parser.add_argument('--alpha', help='coefficient multiply with epsilon (difference between grad of U and grad of W)', type=float, default=1)  
    
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=int, default=64)
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    # expected size of saving
    parser.add_argument('--expected_saving', help='expected number of updates from clients for saving', type=int, default=5)

    # machine environment settings
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help='the number of threads;', type=int, default=1)
    parser.add_argument('--num_threads_per_gpu', help="the number of threads per gpu in the clients computing session;", type=int, default=1)
    parser.add_argument('--num_gpus', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    # the simulating system settings of clients
    
    # constructing the heterogeity of the network
    parser.add_argument('--net_drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    parser.add_argument('--net_active', help="controlling the probability of clients being active and obey distribution Beta(active,1)", type=float, default=99999)
    # constructing the heterogeity of computing capability
    parser.add_argument('--capability', help="controlling the difference of local computing capability of each client", type=float, default=0)
    
    # attacker or not
    parser.add_argument('--attack', help='normal or attack data', type= int, default= 1)
    # clean or not
    parser.add_argument('--clean_model', help='clean_model equals 1 in order to run retrain model and 0 otherwise', type=int, default=0)
    
    # server gpu
    parser.add_argument('--server_gpu_id', help='server process on this gpu', type=int, default=0)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed) 
    torch.cuda.manual_seed_all(123+seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
    train_datas, train_datas_attack, valid_datas, test_data, backtask_data, client_names = task_reader.read_data()
    num_clients = len(client_names)
    print("done")

    # init data of attackers
    s_atk = [0]
    option['attacker'] = s_atk
    
    # import pdb; pdb.set_trace()
    import torchvision.transforms as transforms
    if option['task']=='mnist_cnum25_dist0_skew0_seed0':
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.1307, ), (0.3081,))])
    elif option['task']=='cifar10_cnum18_dist0_skew0_seed0':
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif option['task']=='medmnist_cnum11_dist0_skew0_seed0':
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize(mean=[.5], std=[.5])])
    else:
        raise Exception("Invalid value for task name")

    for cid in range(num_clients):
        if cid in s_atk:
            train_datas_attack[cid] = copy.deepcopy(train_datas_attack[s_atk[0]])
            train_datas_attack[cid].X = transform(train_datas_attack[cid].X)
    
    # init client
    if option['attack'] == 0:
        print('init clients...', end='')
        client_path = '%s.%s' % ('algorithm', option['algorithm'])
        Client=getattr(importlib.import_module(client_path), 'Client')
        clients = [Client(option, name = client_names[cid], train_data = train_datas[cid], valid_data = valid_datas[cid]) for cid in range(num_clients)]
        print('done')
    else:
        print('init clients...', end='')
        client_path = '%s.%s' % ('algorithm', option['algorithm'])
        Client=getattr(importlib.import_module(client_path), 'Client')
        clients = [Client(option, name = client_names[cid], train_data = train_datas_attack[cid], valid_data = valid_datas[cid]) for cid in range(num_clients)]
        print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, utils.fmodule.Model().to(utils.fmodule.device), clients, test_data = test_data, backtask_data = backtask_data)
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_P{:.2f}_TD{:.2f}_GE{:.2f}.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['proportion'],
        option['theta_delta'],
        option['alpha'])
    return output_name

class Logger:
    def __init__(self):
        self.output = {}
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')
            return self.time_buf[key][-1]

    def save(self, filepath):
        ## print accuracy
        print(self.output["test_accs"])
        
        """Save the self.output as .json file"""
        if self.output=={}: return
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        if var_name in [key for key in self.output.keys()]:
            self.output[var_name] = []
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass
    
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
