## DeepPulsarNet
 
This Python 3 program allows searching for pulsars in filterbank files using neural networks.

# Installation

* Install a for your system fitting version of [PyTorch](http://github.com) (Tested on 1.6.0)
* Run 'pip install -r requirements.txt'

For the data preparation and simulation we additionally require:
* [Tempo2](https://bitbucket.org/psrsoft/tempo2/)
* [sigproc](https://github.com/SixByNine/sigproc) with the binaries linked in your PATH
* The [SKA-TestVectorGenerationPipeline](https://github.com/larskuenkel/SKA-TestVectorGenerationPipeline) ('git clone https://github.com/larskuenkel/SKA-TestVectorGenerationPipeline') This version uses v1.0 of the pipeline but is usable in Python 3


# Basic Workflow

* Grab pulsar survey data. One example set of the Parkes Multibeam Survey which already has been sufficiently downsampled can be found [here](https://uni-bielefeld.sciebo.de/s/FPBaOw3Q1rs15BL)\*
* Create a training and noise set using 'create_training_set.ipynb' and 'prepare_noise_set.ipynb' included in ./deeppulsarnet/notebooks
* Train a neural network using 'train_pulsar_net.py'
	* Example command: python train_pulsar_net.py --path simset_training_set_1_noise.csv --path_noise noiseset_noise_sample.csv --name test_model --length 100000
* Make a prediction for a set of observations using 'make_prediction_for_set.ipynb'

The parameters of the network can currently be changed by modifying the .json config files which are given with the '--class_configs' and "-- model_config" options wich use configs included in the ./deeppulsarnet/model_configs folder. Single parameters can be changed with the '--model_parameter' option.

\*Original data:
Lyne, A; Manchester, R; Camilo, F; Bell, J; Sheppard, D; D'Amico, N; Kaspi, V (2012): Parkes observations for project P268 semester 1997AUGT. v3. CSIRO. Data Collection. https://doi.org/10.4225/08/583746ac2c4de