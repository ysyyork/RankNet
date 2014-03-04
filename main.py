'''
Ranknet

Learning to rank using gradient descent. A simplified implementation of the 
algorithm described in http://research.microsoft.com/en-us/um/people/cburges/papers/icml_ranking.pdf
'''

import math
import random
import copy
import time
import matplotlib.pyplot as plt

def plotErrorRate(errorRate):
    '''
    Plot error rate using matplotlib
    '''
    
    plt.plot(errorRate)
    plt.ylabel('Error Rate')
    plt.show()
    
def readDataset(path):
    '''
    Dataset - LETOR 4.0
    Dataset format: svmlight / libsvm format
    <label> <feature-id>:<feature-value>... #docid = <feature-value> inc = <feature-value> prob = <feature-value>
    We have a total of 46 features
    '''

    X_train = [] #<feature-value>[46]
    y_train = [] #<label>
    Query = []   #<query-id><document-id><inc><prob>

    print('Reading training data from file...')

    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query.append(extractQueryData(split))
    print('Read %d lines from file...' %(len(X_train)))         
    return (X_train, y_train, Query)  
            
def extractFeatures(split):
    '''
    Extract the query to document features used
    as input to the neural network
    '''
    features = []
    for i in range(2, 48):
        features.append(float(split[i].split(':')[1]))   
    return features

def extractQueryData(split):
    '''
    Extract the query features from a dataset line
    Format:
    <query-id><document-id><inc><prob>
    '''
    queryFeatures = [split[1].split(':')[1]]
    queryFeatures.append(split[50])
    queryFeatures.append(split[53])
    queryFeatures.append(split[56])
    
    return queryFeatures

def extractPairsOfRatedSites(y_train, Query):
    '''
    For each queryid, extract all pairs of documents
    with different relevance judgement and save them in
    a list with the most relevant in position 0
    '''
    pairs = []
    for i in range(0, len(Query)):
        for j in range(i+1, len(Query)):
            #Only look at queries with the same id
            if(Query[i][0] != Query[j][0]):
                break
            #Document pairs found with different rating
            if(Query[i][0] == Query[j][0] and y_train[i] != y_train[j]):
                #Sort by saving the largest index in position 0
                if(y_train[i] > y_train[j]):
                    pairs.append([i, j])
                else:
                    pairs.append([j, i])
    print('Found %d document pairs' %(len(pairs)))
    return pairs

#The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0/(1.0+math.exp(-x)))

#The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return math.exp(-x)/(pow(math.exp(-x)+1,2))

def random_float(low,high):
    return random.random()*(high-low) + low

#Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0]*J)
    return m

class NN: #Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        #Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden
        self.numOutput = 1

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.activations_input = [1.0]*self.numInputs
        self.activations_hidden = [1.0]*self.numHidden
        self.activation_output = 1.0 #Assuming a single output.
        self.learning_rate = learningRate

        # create weights
        #A matrix with all weights from input layer to hidden layer
        self.weights_input = makeMatrix(self.numInputs,self.numHidden)
        #A list with all weights from hidden layer to the single output neuron.
        self.weights_output = [0 for i in range(self.numHidden)]# Assuming single output
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weights_input[i][j] = random_float(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weights_output[j] = random_float(-0.5, 0.5)

        #Data for the backpropagation step in RankNets.
        #For storing the previous activation levels of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        #print('Propagating input...')
        if len(inputs) != self.numInputs-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations=copy.deepcopy(self.activations_input)
        for i in range(self.numInputs-1):
            self.activations_input[i] = inputs[i]
        self.activations_input[-1] = 1 #Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations=copy.deepcopy(self.activations_hidden)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                #print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.activations_input[i] * self.weights_input[i][j]
            self.activations_hidden[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation=self.activation_output
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.activations_hidden[j] * self.weights_output[j]
        self.activation_output = logFunc(sum)
        return self.activation_output

    def computeOutputDelta(self):
        '''
        Equations [1-3]
        Updating the delta in the output layer
        '''
        
        Pab = 1/(1 + math.exp(-(self.prevOutputActivation - self.activation_output)))
        self.prevDeltaOutput = logFuncDerivative(self.prevOutputActivation)*(1.0-Pab)
        self.deltaOutput = logFuncDerivative(self.activation_output)*(1.0-Pab)
        
    def computeHiddenDelta(self):
        '''
        Equations [4-5]
        Updating the delta values in the hidden layer
        '''
   
        #Update delta_{A}
        for i in range(self.numHidden):
            self.prevDeltaHidden[i] = logFuncDerivative(self.prevHiddenActivations[i])*self.weights_output[i]*(self.prevDeltaOutput-self.deltaOutput)
        #Update delta_{B}
        for j in range(self.numHidden):
            self.deltaHidden[j] = logFuncDerivative(self.activations_hidden[j])*self.weights_output[j]*(self.prevDeltaOutput-self.deltaOutput)

    def updateWeights(self):
        '''
        Update the weights of the NN
        Equation [6] in the exercise text
        '''

        #Update weights going from the input layer to the output layer
        #Each input node is connected with all nodes in the hidden layer
        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weights_input[i][j] = self.weights_input[i][j] + self.learning_rate*(self.prevDeltaHidden[j]*self.prevInputActivations[i]-self.deltaHidden[j]*self.activations_input[i])
                
        #Update weights going from the hidden layer (i) to the output layer (j)
        for i in range(self.numHidden):
            self.weights_output[i] = self.weights_output[i] + self.learning_rate*(self.prevDeltaOutput*self.prevHiddenActivations[i]-self.deltaOutput*self.activations_hidden[i])

    #Removed target value(?)
    def backpropagate(self):
        '''
        Backward propagation of error
        1. Compute delta for all weights going from the hidden layer to output layer (Backward pass)
        2. Compute delta for all weights going from the input layer to the hidden layer (Backward pass continued)
        3. Update network weights
        '''
      
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    def weights(self):
        '''
        Debug: Display network weights
        '''

        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weights_input[i])
        print()
        print('Output weights:')
        print(self.weights_output)

    def train(self, X_train, pairs, iterations=25):
        '''
        Train the network on all patterns for a number of iterations.
        Training:
            Propagate A (Highest ranked document)
            Propagate B (Lower ranked document)
            Backpropagate
        Track the number of misordered pairs for each iteration.
        '''

        errorRate = []
        start = time.time()

        print('Training the neural network...')
        for epoch in range(iterations):
            print('*** Epoch %d ***' %(epoch+1))
            for pair in pairs:
                self.propagate(X_train[pair[0]])
                self.propagate(X_train[pair[1]])
                self.backpropagate()   
            errorRate.append(self.countMisorderedPairs(X_train, pairs))
            #Debug:
            print ('Error rate: %.2f' %errorRate[epoch])          
            #self.weights()
        m, s = divmod(time.time()-start, 60)
        print('Training took %dm %.1fs' %(m, s))
        plotErrorRate(errorRate)

    def countMisorderedPairs(self, X_train, pairs):
        '''
        Let the network classify all pairs of patterns. The highest output determines the winner.
        Count how many times the network makes the wrong judgement
        errorRate = numWrong/(Total)
        '''

        misorderedPairs = 0

        for pair in pairs:
            self.propagate(X_train[pair[0]])
            self.propagate(X_train[pair[1]])
            if self.prevOutputActivation <= self.activation_output:
                misorderedPairs += 1
        
        return misorderedPairs / float(len(pairs))
    
if __name__ == '__main__':

    #Read training data
    X_train, y_train, Query = readDataset('Data/train.txt')
    #Extract document pairs
    pairs = extractPairsOfRatedSites(y_train, Query)
    #Initialize Neural Network
    rankNet = NN(46, 20, 0.001)
    #Train the Neural Network
    rankNet.train(X_train, pairs, 20)
    #Read testset
    X_train, y_train, Query = readDataset('Data/test.txt')
    #Extract document pairs
    pairs = extractPairsOfRatedSites(y_train, Query)
    print('Testset errorRate: ' + str(rankNet.countMisorderedPairs(X_train, pairs)))
    
    