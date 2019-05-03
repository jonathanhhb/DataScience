# DataScience

MNIST TUTORIAL (Keras/CNNs)
===========================

`nrunx --network host --env USER=$USER --env HOME=$HOME --env DISPLAY=$DISPLAY  nvcr.io/idmod/idm_keras:mnist_capable_verified`
`git clone https://github.com/jonathanhhb/DataScience.git`
`cd DataScience/MNIST`
`python3 mnist_tutorial.py --train`
`python3 mnist_tutorial.py --test`
`
What just happened?
This is the famous MNIST 'hello world' of neural networks. At first we train a simple convolutional neural net on the MNIST 
digits dataset. We just use 5 epochs for speed but you can see error gets small fast. The code shows how to do this in 
Keras. The trained network is saved and when you run --test you reload that network and do inferences. Rather than just
measure the accuracy we go digit by digit and find errors and display those images to the user, along with what the NN
thought it was vs what it actually is. You can see that those failures are actually pretty hard for humans to classify also.

---


RANDOM FORESTS
==============

`cd <...>/DataScience/RANDOM_FORESTS/`
`python3 random_forest.py`
`./generate.py > generated.csv`
`python3 random_forest.py generated.csv`

What just happened?
This demo shows Random Forests in Python. This does not require Keras, just SciKit Learn. This first example isn't an actual
dataset but a numerical generated 'dataset' where the output column is literally a mathematical function of the inputs. This
is my version of an RF hello world. Next will be the bostonhousing dataset and then the PIMA indian one.


---

COMING SOON...
==============
* Neural Net Regressor
* Spotify
* Sentiment Analysis
