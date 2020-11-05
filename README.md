# lip6_research
Works on information quantization in NN

## Requirements

See the requirements.txt file
+ Python 3.7

## Getting Started

To train a (basic vgg16) on cifar10 classification, run the folowing :

	python vgg16.py 100 1 5e-2
 (it runs for 100 epochs, with a 1 weight, and a lerning rate of 5e-2)

To insert a compression module (witout buffer) in between vgg layers, run the following :

	python info_flow_eval.py 100 29 64 512 1
 (it runs for 100 epochs, with a 1 weight, and uses a codebook of size (64, 512) inserted after the layer 29)

To insert a compression module (witout buffer) in between vgg layers, with a pretrained VGG where layers are frozen except for compression module, run the following :

	python pretrained_vgg_compression.py 100 29 64 512 1000 2e-4
 (it runs for 100 epochs, with a learning rate of 2e-4, with a 1000 weight, and uses a codebook of size (64, 512) inserted after the layer 29)
 
To insert a compression module (witout buffer) in between vgg layers, without any pretraining, run the following :

	python vgg_compression.py 100 29 64 512 1000 2e-4
 (it runs for 100 epochs, with a learning rate of 2e-4, with a 1000 weight, and uses a codebook of size (64, 512) inserted after the layer 29)

To study the impact of a compression module (witout buffer) in between vgg layers and observe the different losses and training accuracy, with a pretrained VGG where layers are frozen except for compression module, run the following :

	python pretrained_vgg_compression_train_bias.py 100 29 64 512 1000 2e-4
 (it runs for 100 epochs, with a learning rate of 2e-4, with a 1000 weight, and uses a codebook of size (64, 512) inserted after the layer 29)
