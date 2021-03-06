# Feature Ranking

Generating a set of spatial and temporal feature rankings occurs in 3 steps:

1. Training the Backbone Networks
2. Training the Inference Models
3. Ranking the Features in the Models

I describe each step in turn and how it is accomplished using this code-base. All implemenations described here are 
designed for use with the 'Tea Making' application and are described in the context of the VGG-16 CNN Backbone.

## Preparations

The following are steps that are done to prepare for execution of the codebase.

### Dependencies

This code has the following dependencies

```
Python3
PyTorch
PyTorch Geometric
OpenCv
Pillow
Scipy
Numpy
Pandas
Seaborn
```

### Dataset

The Tea Making Dataset can be downloaded from: [Here](https://universitysystemnh-my.sharepoint.com/:f:/g/personal/mah1075_usnh_edu/Eo4yvs-jlt5DtEloFOvGHm8BK9jXIa3ghQvdA30meGouOg?e=FdD5DY)

### Trained Models

Trained models can be located at: [Up on request]

### Parameter Changes

A few changes need to be made to some files to the file "parameter_parser_tea_only.py" on lines 20 and 21.
These should be changed to the home directory and the directory where the TeaMaking files are located respectively.

## Code Execution

### Training the Backbone Model

We begin by training the CNN-backbone. This is accomplished through the 'execute_backbone.py' code. Which will train a 
specific CNN-backbone using the available training dataset. This code is designed to train only the spatial features of 
a backbone model. Because the number of edges generated by the temporal model is an exponential function of the number
of spatial features in the model this architecture assumes a need for a bottleneck (in this work a 1x1 convolution). 
This is applied to the posterior of the convolutional architecture and is user defined (line 25 of execute_backbone.py).

Running the execute_backbone code can be accomplished as follows. Additional commands can be accessed using the --help 
flag: 

```python3 execute_backbone.py <model_name>```

This execution will attempt to train a backbone CNN model (<model_name>) on the Tea Making dataset. Models will be 
generated for each bottleneck listed on line 25 of the execute_backbone.py file. The trained model is placed in a
directory of the name: ./saved_model_<bottleneck_size>/c_<model_name>\_backbone\_<run_id>

In the case of the VGG-16 backbone network the code might be run as follows:

```python3 execute_backbone.py vgg --repeat 3```

And the result would be four directories for the 4 bottlenecks listed on line 25: 8, 16, 32, and 64. The directories 
generated are: 

```saved_models_8/c_vgg_backbone_0
saved_models_8/c_vgg_backbone_1
saved_models_8/c_vgg_backbone_2
saved_models_16/c_vgg_backbone_0
...
saved_models_64/c_vgg_backbone_2
```

Each directory contains the fixed features for the trained model and a file containing the accuracy achieved when 
evaluating the model on a trained dataset: 'results.csv'. The quality of each model can be assessed using the following 
command:

```python3 analysis/model_analyze.py <model_directory>```

This will print out the train and evaluation accuracy of the trained model. The confusion matrix can also be saved
to the model_directory using the --save_cm command which will require a path to the labels in the training dataset.

```
python3 analysis/model_analyze.py saved_models_16/c_vgg_backbone_0 --save_cm
        path: /TeaMaking/frames/train
```

Once a model has been selected it should be moved to a directory titled: base_models_<dataset_name>. For the tea making 
example this directory should be base_models_tea_making. The name of the saved model should be updated in the 
parameter_parser.py file. For VGG-16 this is line 88.

### Training the Inference Models

Once the backbone model has been established it is time to train the inference models (in our work
these were the spatial and temporal inference architectures). This is accomplished through the following command:

```python3 execute.py <model_name> <inference_approach>```

where <inference_approach> is either 'linear_iad' (spatial model) or 'ditrl' (temporal model). This code uses the 
trained features of the fixed backbone model to identify feature presence in the input video. This information
is then passed to either a linear layer or through our temporal feature learning pipeline (refer to text). When conducting
inference using the temporal model the architecture will generate intermediary files (IADs) in the directory where the 
dataset is located in order to expedite learning. The trained models are placed in a directory titled: 
saved_models_<dataset_name>/c_<inference_approach>_<run_id>. This file can be interrogated using the same model_analysis 
code as before.

### Ranking

Finally, the importance of the spatial features in the respective models can be established using the code in 
node_rank.py. 

```python3 node_rank.py <model_directory> <inference_approach>```

This code will locate the trained model and will evaluate the model located there using the Tea Making dataset.
The importance that the model applies to each of the spatial features is output in a file 'importance.csv' within the 
model's directory (where teh results.csv file is located).
