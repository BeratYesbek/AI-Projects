# GOAL OF THIS PROJECT
Predict price of phones using phone software and hardware properties.

### KNN
k-nearest neighbors (k-NN) is a machine learning algorithm that is used for classification and regression. It works by finding the k data points in the training set that are closest to the input data point, and then using those data points to make a prediction.

For example, in a classification problem, the algorithm might find the k training data points that are closest to the input data point and then assign the input data point to the class that is most common among those k data points. In a regression problem, the algorithm might find the k training data points that are closest to the input data point and then use the mean or median of those data points to make a prediction.

k-NN is a simple and effective algorithm that can be used for a variety of tasks, but it can be computationally expensive and may not perform well on large datasets. It is also sensitive to the choice of the value of k, which can affect the accuracy of the model.

<img src="https://user-images.githubusercontent.com/77804034/210006864-bfbce775-c9b4-4f0f-bf3f-d42bb0ddf237.png"/>

### Naive Bayes 

Naive Bayes is a machine learning algorithm that is used for classification. It is based on the Bayes theorem, which is a mathematical principle that allows us to make predictions about the probability of an event based on prior knowledge. In the context of classification, Naive Bayes can be used to predict the class of a data point based on the values of its features.

The "naive" part of the name comes from the assumption that all of the features of the data are independent of one another, which is not always the case in real-world data. Despite this assumption, Naive Bayes can still perform well in many cases and is particularly well-suited for classification problems with large datasets and a large number of features. It is also fast to train and can make predictions quickly, making it a popular choice for tasks such as spam filtering and document classification.


<img src="https://user-images.githubusercontent.com/77804034/210006927-466a99f1-7866-4182-a300-31124e22599d.png"/>

### ANN (Artificial neural networks)
Artificial neural networks (ANNs) are a type of machine learning algorithm that are inspired by the way the brain works. They are made up of layers of interconnected nodes, which are called "neurons." These neurons are organized into input, hidden, and output layers, and they are connected by weights that can be adjusted to improve the performance of the model.

ANNs are trained using a large dataset of input-output pairs, and they learn to make predictions by adjusting the weights between the neurons in the hidden and output layers. This process is called "training," and it is typically done using an optimization algorithm such as stochastic gradient descent. Once an ANN has been trained, it can be used to make predictions on new data by feeding the data through the network and using the weights between the neurons to make a prediction.

ANNs are widely used for a variety of tasks, including image and speech recognition, natural language processing, and prediction tasks in finance and healthcare. They are powerful machine learning tools, but they can be computationally expensive to train and may require a large amount of data to perform well.

<img src="https://user-images.githubusercontent.com/77804034/210007103-c39c79f0-5f5d-4483-abd1-d8df38d61a6b.svg">

### Hill Climbing 
Hill climbing is a heuristic search algorithm that is used to find the maximum or minimum value of a function by iteratively making small moves in the direction that improves the value of the function. It is an optimization algorithm that is used to find the best solution to a problem by starting from an initial solution and repeatedly improving it.

The basic idea behind hill climbing is to start at some initial point and then move in the direction that results in the greatest improvement in the value of the function. The algorithm continues this process until it reaches a local maximum or minimum, which is a point where the function value no longer improves by making small moves in any direction.

Hill climbing can be used to solve a wide range of optimization problems, but it is sensitive to the choice of the initial solution and can get stuck in local optima, which are points that are not the global maximum or minimum but are still considered "good" solutions. To address this issue, other optimization algorithms such as simulated annealing and genetic algorithms have been developed that are less sensitive to the choice of the initial solution and are more likely to find the global optimum.

### Results

<img src="https://user-images.githubusercontent.com/77804034/210007283-edd8c084-6aa2-4cf7-83b2-3bc74da44308.jpg">
