# IETC_conference_2020

## Herpetofauna Species Classification from Images with Deep Neural Network

This repo contains codes for the research work of IETC_2020 conference publication. In this paper, In the paper work, a Convolutional Neural Network (CNN) architecture has been suggested to classify any two species automatically. 

As an initial experiment, a binary CNN network has been trained and validated to classify snakes, and toads/frog with 9,000 samples collected from various internet public databases. The model evaluation achieved 76% test accuracy on average for the test data that supports the prospects for the recommended model. 

A callback function has been applied to the model that stops the training procedure after reaching a particular accuracy/loss score. In thia ripository contains two seperate code: with and without callback function. 

#### To read full paper click [here](https://ieeexplore-ieee-org.libproxy.txstate.edu/document/9249141)

### Libraries:
	'cv2 
	tensorflow
	keras
	scikit-learn 
	numpy
	matplotlib 
	h5py
	itertools
	datetime'
