# U-Net in tensorflow 2.1
This is an implementation of the U-Net architecture proposed by [Ronneberger
 et al.](https://arxiv.org/abs/1505.04597) in tensorflow 2.1.
 
## Components

### U-Net
The core of this project is the U-Net implementation, which can be found in
`model.py`. It is implemented as a model factory with the keras functional API.

### Data processing
There are some utilities which can be used for data loading and fold
splitting  (`Dataset` class) and data augmentation (`DataAugmenter` class).
These can be found in `data.py`.
The dataset class expects images and corresponding labels to be in png format.
Images should be in a directory separate from the labels, these locations can
be set on initialising the dataset class. Labels should have the same
filename as the corresponding images.
 
### Losses
An implementation of the dice loss, as well as the weighted binary cross
entropy, and a combination
of both can be found in `losses.py`.
  
## Set Up
Clone the repository and make sure you have the requirements needed by running
```
pip install -r requirements.txt
```

Ensure that everything is running as expected by using unittest (in the
 project directory):
```
python -m unittest discover tests
```

## Usage
`experiment.py` shows how to set up a simple training experiment using the
keras training API. To start an experiment, set your image and label
directories in the file, navigate to the project directory and simply run
```
python experiment.py
```
