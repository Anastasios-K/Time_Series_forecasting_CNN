## Time Series forecasting with 1-D CNN and Exponential learning rate with restart points

## Contents
 - [Summary](#summary)
 - [Key features](#key-features)
 - [Requirements: package installation](#requirements-package-installation)
 - [Requirements: dataset format](#requirements-dataset-format)
 - [User guidelines](#user-guidelines)
 - [Computational resources & Execution time](#computational-resources-and-execution-time)
 - [Future development](#future-development)
 - [References](#references)
 
## Summary
the summary is in progress...
 

## Key features

1. Integration of 1D CNN and Sliding Window.
2. Exponential learning rate with restart point.
3. Experimentation with 4 learning rate strategies.
4. Best model selection considering 1280 variants.
5. Reproducibility 100%.
6. Coverage 100%.
7. Automatic recovering of the training process, in the case of unexpected interruption.

## Requirements: package installation
conda version: 4.8.3

 - python=3.7.9
 - tensorflow-gpu=2.1.0
 - keras=2.3.1-0
 - scikit-learn=0.23.2
 - scipy=1.5.2
 - pandas=1.1.3
 - numpy=1.19.2
 - matplotlib=3.3.2
 - tqdm=4.50.2
 - openpyxl=3.0.5

## Requirements: dataset format
The given dataset should have the following attributes:
 - Local Date
 - Local Time
 - Close
 - Open
 - Low
 - High
 - Volume

If extra attributes are included, they are automatically removed during the data preparation phase.
> **WARNING**: If any of the above attributes is missing, the process will not run properly.
 
## User guidelines
 - To execute the whole process:
    - Run the main.py
    - the execution also includes:
        1. preparation of the saved_models and saved_reports directories in the root path.
        2. saving plots, learning rate logs and exploration reports in the root path.
        3. saving the best weights of each trained model in the saved_models directory.
        4. saving the training progress report of each trained model in the saved_reports directory.
 
 - To change a general parameter:
    - Change the value of the corresponding parameter
     in the yaml file [parameters_general](parameters_general.yaml).
 
 - To change a hyper parameter:
    - Change the value of the corresponding parameter
     in the yaml file [parameters_hyper](parameters_hyper.yaml).
 
 - To increase or decrease the possible values of a hyper parameter:
    - Add or remove a specific value in the corresponding parameter
    in the yaml file [parameters_hyper](parameters_hyper.yaml).
    - Keep the "list" form, even if a hyper-parameter values are reduced to 1.
 
 - To disable the 3rd convolution layer:
    - Remove "True" from the hyper-parameter called "third_convolution_layer_added".
    - Keep only "False" in the list.
    - The training process will automatically skip all possible model variants 
    which are associated with the activation of the 3rd convolution layer.
    

> **WARNING**: Changing the parameters key may harm the execution

## Computational resources and Execution time
**CPU:** Intel Core i5-9500 @ 3.00 Ghz, 6 cores  
**GPU:** Nvidia GeForce RTX 2070 Super  

**Execution time:** 32 hours approximately
 - Using the following:
    1. the default [hyper-parameters](parameters_hyper.yaml)
    2. the [GSK](data/GSK%20per%20min.csv) dataset

## Contribution
Pull requests are more than welcome.  
However, if you consider major changes, **PLEASE** open an issue first.

## Future development
- Creating Docker
- Enabling Google Colab execution
- Integrating Tensorboard
- Enabling direct use of the best model

## References
 - [Polynomial learning rate policy with warm restart for deep neural network](https://ieeexplore.ieee.org/abstract/document/8929465)
 - [Cyclical Learning Rates for Training Neural Networks](https://ieeexplore.ieee.org/abstract/document/7926641)