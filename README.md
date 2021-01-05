# NeuralNetworkFromScratch
My implementation of a Feed Forward Neural Netwrok using numpy. 

The implementation is heavily inspired by Michael Nielson's, I cannot stress how great, [book](http://neuralnetworksanddeeplearning.com/index.html). I have added some features that are not in his implementation, such as vectorized computation, which reduces training time 5 times, and a method to visualize the learned weights as images.

## Visualizing first layer weights
Let's have a look at what it means for a neural network to learn. Most of the abstractions NNs learn are hard to interpret for humans, but there is a whole movement towards intepretability that has come up with quite ingenious ways of visualizing what neural networks learn. Most of that research is focused on convolutional neural networks, but I have implemented a simple way to visualize the weights of a FFN. This approach is limited to just the weights in the first layer, as only for those can we make sense of the ordering of the relative positions.

### untrained weights
As we can see, in an untrained network the weights in the first layer are random.

![alt text](https://github.com/MichalPitr/NeuralNetworkFromScratch/blob/main/imgs/untrained.png)
### trained weights

But once we train the network, we notice some emerging patters. We can think about these 9 images as filters or feature detectors that are passed to subsequent layers to make more nuanced decisions. (Note: The network architecture is simply [784, 9, 10]. I chose 9 neurons in the hidden layer as it makes it particularly easy to fit the images here.)


![alt text](https://github.com/MichalPitr/NeuralNetworkFromScratch/blob/main/imgs/trained_weights.png)
