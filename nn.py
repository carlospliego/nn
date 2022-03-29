import numpy as np
from PIL import Image
import glob

class NeuralNetwork():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((100, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

def load_file_bits(data):
    files = sorted(glob.glob(data))
    data = []
    for file in files: 
        # load the bitmap
        im = Image.open(file).convert('L')
        p = np.array(im)
        row = []
        for r in p:
            for c in r:
                if c==0:
                    row.append(1)
                else:
                    row.append(0)
        data.append(row) 
    return data

# should take in a 
def load_testing_data():
    return load_file_bits("/src/testing_files/*")

def load_training_data():
    return load_file_bits("/src/training_files/*")

if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    training_data = load_training_data()
    training_inputs = np.array(training_data)

    training_outputs = np.array([[
        1,1,1,1,1, # Circle
        0,0,0,0,0 # Not Circle
    ]]).T

    neural_network.train(training_inputs, training_outputs, 15000)
    testing_data = load_testing_data()

    print('Classification', neural_network.think(np.array(testing_data)))
