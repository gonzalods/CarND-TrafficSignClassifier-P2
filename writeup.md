# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1_1]: ./images/train_distribution.png "Train Distribution"
[image1_2]: ./images/validation_ditstribution.png "Validation Distribution"
[image1_3]: ./images/test_distribution.png "Test Distribution"
[image2_1]: ./images/grayscaled.png "Grayscaling"
[image3_1]: ./images/augmented.png "Augmented"
[image3_2]: ./images/augmented_distribution.png "Augmented Distribution"
[image4_1]: ./images/lenet-5-nodrop.png "Lenet-5 Wihtout Dropout"
[image4_2]: ./images/lenet-5-drop.png "Lenet-5 Wiht Dropout"
[image4_3]: ./images/lenet-5-32.png "Lenet-5 32x64 Wiht Dropout"
[image4_4]: ./images/conv3.png "Final Conv 3 Layers"
[image5_1]: ./images/27.png "Traffic Sign Children crossing"
[image5_2]: ./images/21.png "Traffic Sign Double courve"
[image5_3]: ./images/38.png "Traffic Sign Keep right"
[image5_4]: ./images/2.png "Traffic Sign Speed limit 50 km/h"
[image5_5]: ./images/6.png "Traffic Sign End of speed limit 80 km/h"
[image6_1]: ./images/prob_new_img_27.png "Model Certantly Children crossing"
[image6_2]: ./images/prob_new_img_21.png "Model Certantly Double curve"
[image6_3]: ./images/prob_new_img_38.png "Model Certantly Keep right"
[image6_4]: ./images/prob_new_img_2.png "Model Certantly Speed limit 50 km/h"
[image6_5]: ./images/prob_new_img_6.png "Model Certantly End of speed limit 80 km/h"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set.

I used the python and numpy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the 43 different traffic signs. All three data sets (train, validation and test) have a similar poor balanced distribution.

![alt text][image1_1] ![alt text][image1_2] ![alt text][image1_3]

### Design and Test a Model Architecture

#### 1. Preprocess Image Data

As first stept, I applied an aproximated normalization of data (fixed value of , in final model mean and standar deviation), that meant an improvement in the validation accuracy of approximately  2.5 points (88.9% - 91.1%).

The next step in the preprocessing of the data was to convert the data to grayscale and to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) [explanation here](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)  . That meant a great improvement of more than 4 points in the validation accuracy (from 91.1% to 95.2%).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2_1]

Finnaly, and due to the poor balance in the distribution of de original data (although very similar in the tree data sets),
I generated additional data for trainning. To add new data to the data set, I used techniques of shift, rotate, scale and brightness in
the original data. This meant a better balance in the distribution of the data but not completely uniform. Perhaps this is why the
improvement in accuracy only accounted for a few decimas in the validation data (95.2 - 95.8), decreasing in the case of the training data.
The number of the augmented data set is

Here is an example of an original image and an augmented image:

![alt text][image3_1]

The distribution of the augmented data is

![alt text][image3_2]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image                       |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x46 	|
| RELU					|												|
| Dropout               | keep prob 0.5                                 |
| Max pooling	      	| 2x2 stride,  outputs 14x14x46 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x92   |
| RELU					|												|
| Dropout               | keep prob 0.5                                 |
| Max pooling	      	| 2x2 stride,  outputs 5x5x92   				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 5x5x184     |
| RELU					|												|
| Dropout               | keep prob 0.5                                 |
| Max pooling	      	| 1x1 stride,  outputs 4x4x184   				|
| Fully connected		| 2944x1000                                     |
| RELU                  |                                               |
| Dropout               | keep prob 0.5                                 |
| Softmax				| wiht one-hot codification                     |


#### 3. Model Training
Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following elements:

* Loss function: cross-entropy.
* Optimizer: Adam optimizer.
* Learning rate: 0.0001
* Batch size: 128
* Epochs: 30

I have used the Adam optimizer because it has been demostrated that it works well in practice and compares favorably to other stochastic optimization methods ([explanation here](https://arxiv.org/abs/1412.6980)). Adam optimizer is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.

#### 4. Solution Approach

I started with a LeNet-5 model in RGB with 6 and 16 filters in the convolutional layer, batch size of 128, 10 Epochs and a learning rate of 0.001 . This solution gave a distant performance to the objective of the project (Validation accuracy 88.4%).

To improve the performance I used the preprocessing techniques mentioned above. This improved the performance and exceeded the precision goal of the project but it suffered from certain overfitting (Validatión accuracy 95.8% - Train accurancy 99.5%).

![alt text][image4_1]

To correct the overfitting I used dropout on all the layers with a value of 0.5. At the same time I increased the number of epochs to 20. This gave as result a validation accuracy of 96.7% and a training accuracy of 98.4%.

![alt text][image4_2]

To keep improving the performance, I increased the complexity of LeNet-5 a bit by using a depth of 32 and 64 in the conv layers. The number of times to 25 and the learning rate to 0.0001 are also updated. This gave as result a validation accuracy of 97.5% and a training accuracy of 99.4%.

![alt text][image4_3]

Finally, get to the final model

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.0%
* test set accuracy of 97.5%

![alt text][image4_4]


### Test a Model on New Images

#### 1. Acquiring New Images.

These are five German traffic signs that I found on the web:

![alt text][image5_1] ![alt text][image5_2] ![alt text][image5_3]
![alt text][image5_4] ![alt text][image5_5]


#### 2. Performance on New Images
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The results of the prediction were:

| Image			             |     Prediction	        					|
|:--------------------------:|:--------------------------------------------:|
| Children crossing          | Children crossing            				|
| Double curve               | Double curve									|
| Keep right                 | Keep right									|
| Speed limit (50km/h)	     | Speed limit (50km/h)			 				|
| End of speed limit (80km/h)| End of speed limit (80km/h)                  |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.5%.

#### 3. Model Certainty - Softmax Probabilities

The certainty model of the five images selected from the web is as follows

![alt text][image6_1]

![alt text][image6_2]

![alt text][image6_3]

![alt text][image6_4]

![alt text][image6_5]
