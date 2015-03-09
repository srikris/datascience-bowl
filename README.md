Python start guide for data science bowl
----------------------------------------

The following set of scripts should be a good way to get started with
[Convolution Neural
Networks](http://en.wikipedia.org/wiki/Convolutional_neural_network).  It uses
a [GraphLab-Create's deep
learning](https://dato.com/learn/userguide/#neural-net-classifier) which is
based on CXXNet. 

* **Setup time**: ~2 mins
* **Train time**: ~20 mins on a GPU (it could take much longer on a CPU)
* **Validation score**: 0.76
* **Leaderboard score**: 0.97


Solution
--------

Here is a quick summary of the submission:

* Load images into an SFrame (scalable dataframe).
* Use [Pillow](https://pypi.python.org/pypi/Pillow/) to augment the data with
  rotations with angle 90, 180, and 270.
* Setup a simple deep learning architecture (based on
  [antinucleon](https://github.com/antinucleon/cxxnet/blob/master/example/kaggle_bowl/bowl.conf]))
* Create a "fair" train, validaiton split to make sure the classes are balanced.
* Train a deep learning model.
* Evaluate the multi-class log loss score.
* Save the predictions in Kaggle's format into a submission file called "submission.csv".


Install
-------

**CPU instructions**
```
pip install -r requirements.pip
```

**GPU instructions**
```
pip install -r requirements-gpu.pip
```

Data
-----
Let us assume that you have the data downloaded into two folders called train 
and test. You can do that as follows:

```
wget https://www.kaggle.com/c/datasciencebowl/download/train.zip
wget https://www.kaggle.com/c/datasciencebowl/download/test.zip
unzip train.zip
unzip test.zip
```

Make submission
---------------

Now run the following script. The script will create a submission file. It 
could take around 1 hour depending on how many interations you perform. The
network can train at around 5k images a second.

```
python make_submission.py
```
