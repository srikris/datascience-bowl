Python Start Guide for Deep Network
====================================

The following set of scripts should be a good way to get started with CNNs
(Convolution Neural
Networks)[http://en.wikipedia.org/wiki/Convolutional_neural_network].  It uses
a [GraphLab-Create's deep
learning](https://dato.com/learn/userguide/#neural-net-classifier) which is
based on CXXNet. 

**Setup time**: < 2 mins
**Train time**: < 20 mins on a GPU (it could take much longer on a CPU)
**Validation score**: 0.78
**Leaderboard score**: 0.78


Summary of things done:
* Load images into a dataframe.
* Use [Pillow](https://pypi.python.org/pypi/Pillow/) to augment the data with
  rotations with angle 90, 180, and 270.
* Setup a simple deep learning architecture (based on (antinucleon)[https://github.com/antinucleon/cxxnet/blob/master/example/kaggle_bowl/bowl.conf])
* Create a "fair" train, validaiton split
* Evaluate the model
* Make a submission file called "submission.csv" 


CPU Setup
--------------
pip install -r requirements.pip

GPU Setup
--------------
pip install -r requirements-gpu.pip

Download data
--------------
Let us assume that you have the data downloaded into two folders called train 
and test. You can do that as follows:

wget https://www.kaggle.com/c/datasciencebowl/download/train.zip
wget https://www.kaggle.com/c/datasciencebowl/download/test.zip
unzip train.zip
unzip test.zip

Make submission
---------------
python make_submission.py
