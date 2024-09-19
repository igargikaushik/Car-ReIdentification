## Car Re-Identification

- traditional: algorithm

- simaese: network picture search

- deep_hash_retrieval: Image Retrieval


### paper:
The paper calculates distance. Euclidean distance and Hamming distance are very fast, basically at the US level.
The hash retrieval network has one more fully connected layer than the classification network (AlexNet used in the paper), so the network parameters before the fully connected layer can be pre-trained, and the parameters after the hidden layer are randomly set.
Increasing the number of hidden layer nodes does not necessarily result in better results.

### idea:
It was found that rough screening only compares hash values, so a more complex network will only affect the time to generate binary values, but will not affect the comparison process.
I previously thought about adding the twin network, but found a retrieval problem. The images to be retrieved must be of the same type as the database, so I couldn't bring in different types.
How to improve accuracy? What should I do if I can’t retrieve it? Or reduce the error rate to very low.
It is impossible to retrieve all the number of pictures. You can only set a Hamming distance to retrieve the pictures within the distance.
If the coarse retrieval threshold is set to 2, we can no longer rely on coarse retrieval to distinguish similarities, and can only use another method to conduct fine retrieval.
When the Hamming distance is the same, the Euclidean distance is the same, so it seems that the Hamming distance can be used for sorting.

### dataset:
Features of the vehicle data set: The vehicle has been cut out and there is little background interference.

### implementation steps:
#### Rough screening:
After adding the hidden layer and adding the activation function output, set a threshold so that the output value of each node is 0 or 
Find the Hamming distance of the node output (the number of different codes between the two codes), and set a threshold so that there is a range that is similar to the detected image.
#### Fine filter:
Calculate the Euclidean distance on the output of the hidden layer (not binary) and find that there is no distinction between the same cars.
#### github code:
Retrieve image hidden layer nodes.

### code:
Test Category
13/9 alexnet 63epoch 97% of the training set and 76% of the test set showed overfitting in the training set
13/9 alexnet 48eopch training set 98% test set 77% Add dropout based on the previous weights
16/9 resnet34 120epoch training set 99% test set 79%
16/9 resnet34 79epoch training set 98% test set 79% increase scale rotation, brightness, contrast, saturation changes
16/9 resnet34 24epoch training set 98% test set 79% Change the mean and variance to your own data.
16/9There is a problem with resnet34 transform data conversion. I changed it to 80%. Let’s go to 80 first.
18/9 resnet50 80%
19/9 resnet50 100% I just found that the data of two categories in the test category were misclassified, resulting in the accuracy rate not reaching 80% before.

### test retrieval:
17/9 Querying 1,000 pictures in a 9,000-image database took 2.86S. The speed is still acceptable.
19/9 The topk retrieval accuracy [0.8 0.79946 0.79973 0.79995333 0.7995925 0.718744]-》[1, 50, 100, 600, 800, 1000] is largely determined by the performance of the network
19/9 The Hamming distance is set to 10. The query time for 1000 query pictures and 9000 pictures in the database is 4.38s. 
