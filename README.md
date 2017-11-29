# Noise-Layer-Implemantation-on-AlexNet

This code implements a training in TensorFlow for noisy data by utilizing a linear noise layer which is proposed in [Training Convolutional Networks with Noisy Labels](https://arxiv.org/abs/1406.2080). 

AlexNet implementation and the model is directly taken from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ and some modifications are done on this code. Moreover, in order to feed the train and validation batches, "datagenerator.py" is taken from Frederik Kratzert's [repo](https://github.com/kratzert/finetune_alexnet_with_tensorflow).

In order to prepare the data files, "createDataFiles.m" script can be used.

The general process is the following:
- Fine-tune the network up to some point without using the noise layer and save the updated weights from fc-7 and fc-8 (named in the code as "base_snapshot.pkl"). 
- After initializing the layers by using these weights, start to fine-tune the network again but this time use the linear noise layer as well.

The loss graphs of train and validation sets can be observed from TensorBoard.

For the experimental results, the details of the implementation and further information, you can read this paper (coming soon).
