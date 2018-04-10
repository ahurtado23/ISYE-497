import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import collections
import re, os, cv2
from glob import glob

# We know that MNIST images are 28 pixels in each dimension.
img_hieght = 22

img_width = 30
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_hieght * img_width

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_hieght, img_width)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 62


   # It returns a bit array (of 0s and 1) of ‘num_classes’ length, with only a 1 in the current folder    
   #      position and 0s in all other folder position.
   # In other words, indexes all folders and says which one it’s on. 
def create_argmax(folder_name):
    index = int(re.search('Sample(.*)', folder_name).group(1)) - 1
    arr = np.zeros(num_classes)
    arr[index] = 1
    return arr


# It returns the Gray-scale version of the input image flatted as an array of (img_height * img_width) length.
def cvt_img2np(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(660).astype(np.float32, copy=False)



def get_dataset():
    DATASET_LOCATION = '/var/www/python/ML/hcr/English/Hnd/Img'
   # The following initializes the HCR structure (HandwrittenCharacterRecognition structure), which   
   #    wraps three arrays: data, labels and cls.
   #   This is the algorithm’s core structure...
    hcr = {'data':[], 'labels':[], 'cls':[]}

    # For each SampleXXX folder in the DATASET_LOCATION ... 
    for folder in glob(DATASET_LOCATION + '/*'):
            if os.path.isdir(folder):       
                # For each PNG image in current SampleXXX folder …   
                for img_path in glob(folder + '/*.png'):
                    #  1- to append the Grey converted version of the image flatted as an array of 
                            (img_height * img_width) length, 
                            to the end of ‘data’ array of HCR structure ...
                    hcr.get('data').append(cvt_img2np(img_path))

                    #  2- to append a bit array (of 0s and 1) of ‘num_classes’ length, with only a 1 in the 
                    #      current folder position and 0s in all other folder position,
                    #      to the end of ‘labels’ array of HCR structure ...
                    hcr.get('labels').append(create_argmax(folder))

                    #  3- to append the index of current folder position to the end of ‘cls’ array of HCR 
                    #      structure (note the index must be into [0, num_classes] range)…. 
                    hcr.get('cls').append(int(folder[-3:]) - 1)

    print('read completed')

    # shuffle data:  Generates a Permutation of the data array contained in HCR structure (note that 
    #     hcr.labels and hcr.cls arrays are both permuted too to maintain consistency with permuted 
    #     hcr.data array) ...
    permutate = np.random.permutation(len(hcr.get('data')))
    return {k1:v1[permutate] for k1, v1 in {k: np.array(v) for k, v in hcr.items()}.items()}



# Creates an HCR structure as a Permutation of the Images 
#   included in the DATASET_LOCATION...
hcr = get_dataset()

ran_iterations = 0
def get_next_batch(count):
    global ran_iterations
    data = (np.squeeze(hcr.get('data')[ran_iterations: ran_iterations + count]), np.asarray(hcr.get('labels')[ran_iterations: ran_iterations + count]))
    ran_iterations += count
    return data

 
img = hcr.get('data')[0]




#get this dataset  and turn into 2 text files one test dataset and one train dataset
#modify and format the dataset to fit the text files in example

#ale code starts here




import sys
from PIL import Image
import cv2

FORM_PATH = 'DATASET_LOCATION'

def crop_on_border(img_path):
    img = Image.open(img_path)
    nonwhite_positions = [(x,y) for x in range(img.size[0]) for y in range(img.size[1]) if img.getdata()[x+y*img.size[0]] != 255]
    rect = (min([x for x,y in nonwhite_positions]), min([y for x,y in nonwhite_positions]), max([x for x,y in nonwhite_positions]), max([y for x,y in nonwhite_positions]))
    return img.crop(rect) #.save('out.jpg')
 
def create_binary_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # df = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,100)
    mask_inv = cv2.bitwise_not(mask)
    #denoised = cv2.fastNlMeansDenoising(mask_inv, None, 7, 21, 30)
    return cv2.imwrite('binary_image.jpg', mask)
    


create_binary_image(FORM_PATH)
crop_on_border('binary_image.jpg').save('out.png')


f= open("mnist_train_100.txt","w+") #opens and creates a text file


for i in range(10):  #writes data into a file figure out how to insert binary image into text
     f.write("This is line %d\r\n" % (i+1))


     f.close() #closes file when done





#test data set EDIT HERE NOT DONE
session = tf.Session()
session.run(tf.global_variables_initializer())

hcr_test_data = get_dataset()
feed_dict_test = {
                  x: hcr_test_data.get('data'),
                  y_true: hcr_test_data.get('labels'),
                  y_true_cls: hcr_test_data.get('cls')
                 }

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
 
 #Function for performing a number of optimization iterations so as to gradually improve the weights and biases of the model. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.










#ale code ends here







#neuralnetwork code to machine learn the dataset 

# coding: utf-8

# In[2]:


#Alex Hurtado 
#ISYE 497 
#DIY NEURAL NETWORK

import numpy
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.learningrate = learningrate

        # weights from input to hidden
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        # weights from hidden to output
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes,self.hnodes)))

        self.activation_function= lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # Calculate input * weight and use sigma function to get output
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Same for hidden -> output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # calculate error (target - final_output which is the current value)
        output_errors = targets - final_outputs

        # hidden layer error. weights^T from hidden_output * errors_output
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update values between hidden and output layer
        self.who += self.learningrate * numpy.dot((output_errors*final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the input -> hidden layer weights
        self.wih += self.learningrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        # Convert the input data into an array
        inputs = numpy.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layer (Input for Hidden layer)
        hidden_inputs = numpy.dot(self.wih,inputs)

        # calculate hidden output
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate final input
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784 
hidden_nodes = 150
output_nodes = 10

learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train_100.txt","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# training
for e in range(5):
    for record in training_data_list:
        all_values = record.split(",")
        # scale and shift
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
test_data_File = open("mnist_test_10.txt","r")
test_data_list = test_data_File.readlines()
test_data_File.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99)+0.1

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)
    print(label, "networks answer")

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)


# In[24]:


# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
  


# In[25]:


# following generates an array of values selected at random between 0 and 1

import numpy

numpy.random.rand(3, 3)


# In[26]:


numpy.random.rand(3, 3) - 0.5

