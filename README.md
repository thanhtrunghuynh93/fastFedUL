# Fast-FedUL
Repo for 'Federated Unlearning with Provable Skew Resilience: A Training-Free Approach'

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