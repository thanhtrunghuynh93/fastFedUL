# Fast-FedUL

[![arXiv](https://img.shields.io/badge/arXiv-2405.18040-b31b1b.svg)](https://arxiv.org/abs/2405.18040)

Repo for 'Fast-FedUL: A Training-Free Federated Unlearning with Provable Skew Resilience'

Other baselines can be found at: https://github.com/tamlhp/awesome-machine-unlearning

## Citation

Please read and cite our paper: [![arXiv](https://img.shields.io/badge/arXiv-2405.18040-b31b1b.svg)](https://arxiv.org/abs/2405.18040)

>Huynh, T.T., Nguyen, T.B., Nguyen, P.L., Nguyen, T.T., Weidlich, M., Nguyen, Q.V.H. and Aberer, K., 2024. Fast-FedUL: A Training-Free Federated Unlearning with Provable Skew Resilience. arXiv preprint arXiv:2405.18040.

```
@article{huynh2024fast,
  title={Fast-FedUL: A Training-Free Federated Unlearning with Provable Skew Resilience},
  author={Huynh, Thanh Trung and Nguyen, Trong Bang and Nguyen, Phi Le and Nguyen, Thanh Tam and Weidlich, Matthias and Nguyen, Quoc Viet Hung and Aberer, Karl},
  journal={arXiv preprint arXiv:2405.18040},
  year={2024}
}
```

## QuickStart

**First**, run the command below to get the ARDIS dataset (which is used as backdoor data for MNIST dataset):

```sh
# change to the ARDIS_DATASET_IV folder
cd customdata/ARDIS_DATASET_IV
# unrar ARDIS_DATASET
unrar e ARDIS_DATASET_IV.rar
# return to the root folder
cd ...
```

**Second**, run the command below to get the splited dataset MNIST:

```sh
bash gen_data.sh
```
The splited data will be stored in ` ./fedtask/mnist_cnum25_dist0_skew0_seed0/attack.json`.

**Third**, run the command below to quickly run the experiment on MNIST dataset:

```sh
# all parameters are set in the file run_exp.sh
bash run_exp.sh
```
The result will be stored in ` ./fedtasksave/mnist_cnum25_dist0_skew0_seed0/R51_P0.30_alpha0.07/record/history51.pkl`.

**Finally**, run the command below to return accuracy:

```sh
# return accuracy
python test_unlearn.py
# Main accuracy: ...
# Backdoor accuracy: ...
```
