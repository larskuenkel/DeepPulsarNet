## DeepPulsarNet
 
This Python 3 program allows searching for pulsars in filterbank files using neural networks.

[Dissertation: Lars Künkel - Detecting Pulsars with Neural Networks](https://pub.uni-bielefeld.de/record/2965218)

[Paper: Künkel et al. -  Detecting pulsars with neural networks: a proof of concept ](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1111K/abstract)

# Installation

* Install a for your system fitting version of [PyTorch](http://github.com) (Tested on 1.6.0)
* Run `pip install -r requirements.txt`


# Basic Workflow

* Grab pulsar survey data. One example set of the Parkes Multibeam Survey which already has been sufficiently downsampled can be found [here](https://zenodo.org/records/15399789)\*
* Create a training and noise set using `create_training_set.ipynb` and `prepare_noise_set.ipynb` included in ./deeppulsarnet/notebooks
* Train a neural network using `train_pulsar_net.py`
	* Example command: `python train_pulsar_net.py --path simset_training_set_1_noise.csv --path_noise noiseset_noise_sample.csv --name test_model --length 100000`
* Make a prediction for a set of observations using `make_prediction_for_set.ipynb`

The parameters of the network can currently be changed by modifying the .json config files which are given with the `--class_configs` and `-- model_config` options wich use configs included in the ./deeppulsarnet/model_configs folder. Single parameters can be changed with the `--model_parameter` option.

\*Original data:
Lyne, A; Manchester, R; Camilo, F; Bell, J; Sheppard, D; D'Amico, N; Kaspi, V (2012): Parkes observations for project P268 semester 1997AUGT. v3. CSIRO. Data Collection. https://doi.org/10.4225/08/583746ac2c4de


# Tutorial

* When working with the dockerfile the data loader will most likely run into memory issues which can be fixed by adding `--shm-size 8G` to your `docker run` command.
* `cd tutorial`
* `python 0_create_pmps_dataset.py`
* `python 1_create_simulations.py`
* `python 2_create_targets.py`
* `bash 3_train_network.sh`
* The parameters for the training or the parameters of the simulation set can be changed to increase performance.
* `bash 4_test_network.sh`
* If the Pulsar Prediction value is above 0.5 the network thinks that there is a real pulsar in the data. Half of the test samples contains known pulsars.
