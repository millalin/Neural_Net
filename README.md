# Neural_Network

Neural Network for predicting how an “x” parameter value
change affects the average user count in file [users.csv](https://github.com/millalin/Neural_Net/blob/529a804a128f84f719b7869cc5fc280e7a9a00de/users.csv)

Target cell L1547777

Used libraries are Tensorflow Keras and in Keras Dense and Sequential. For visualisation Matplotlib.

Beacause given data is quite small, whole data is used for training and testing neural network. It has not been divided into several datasets (training dataset, validation dataset and testing dataset) at this point so overfitting and unbiased evaluation is not tested. The model is fitted on training dataset, that uses x parameters values and hour info as input vector and target cell L1547777 user count values as output. You can see effect of adjusting epochs and some results [here](https://github.com/millalin/Neural_Net/blob/529a804a128f84f719b7869cc5fc280e7a9a00de/analysis/doc.md).

 

