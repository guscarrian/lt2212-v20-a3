import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch import autograd
# Whatever other imports you need
import random
import sklearn.metrics as metrics

# You can implement classes and helper functions here too.
class AuthorPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size=0, activation_fn=None):

        super(AuthorPerceptron, self).__init__()

        non_linear_map = {"ReLU":nn.ReLU(),
                          "Tanh":nn.Tanh()}
        print(f"activation_fn: {activation_fn}")
        #Create the layers
        #The size of the first layer is indicated as a parameter
        #The output depends on the hidden layer: hidden_size if the parameter is not 0; 1 otherwise
        layer1_out_size = hidden_size if hidden_size != 0 else 1
        self.layer1 = nn.Linear(input_size, layer1_out_size)
        #layer2 take a value if the activation_fn argument is not None. We use ReLU or Tanh
        self.layer2 = None if activation_fn is None else non_linear_map[activation_fn]

        #In a similar fashion, the hidden layer is created if the hidden_size is not 0
        self.hidden_layer = nn.Linear(hidden_size, 1) if hidden_size != 0 else None
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        out = self.layer1(data)
        if self.layer2 is not None:
            out = self.layer2(out)
        if self.hidden_layer is not None:
            out = self.hidden_layer(out)

        return self.sigmoid(out)

    def train(self, data, batch_size, number_epoch, iterations, lr = 0.01):
        loss_function = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(number_epoch):
            for i in range(iterations):
                samples = get_sample(data, batch_size)
                
                tensors, labels = samples[0], samples[1]
                #tensors, labels = autograd.Variable(torch.FloatTensor([tensor]), requires_grad=True), autograd.Variable(torch.FloatTensor([[label]]))

                for tensor,label in zip(tensors, labels):
                    #Pass the data to the layers of the network
                    output = self.forward(tensor)
                    loss = loss_function(output, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


    def test(self, data, batch_size):
        samples = get_sample(data, batch_size)

        tensors, labels = samples[0], samples[1]
        predictions = []
        for tensor, label in zip(tensors, labels):
            output = self.forward(tensor)
            for value in output:
                predictions.append(get_prediction(value))
        
        get_statistics(labels, predictions)

def get_prediction(value):
    return 1 if value > 0.5 else 0

def get_statistics(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='weighted')
    recall = metrics.recall_score(labels, predictions, average='weighted')
    f1 = metrics.f1_score(labels, predictions, average='weighted')
    
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-measure: ', f1)
                
def get_random_pair_of_author(data, same_author):
    #Select two random authors
    author1 = random.choice(data.Author)
    author2 = random.choice(data.Author)
    #If the argument same_author is 0, keep selecting a random author until the author2 is different from the first one
    if same_author == 0:
        while author1 == author2:
            author2 = random.choice(data.Author)

    return author1, author2

def get_random_tensor(data, author):
    #Get a sample from the data where the author is the same as the indicated author
    #Getting a random row
    author_sample = data[data["Author"] == author].sample(n=1)
    #Removeing the Author and Train/Test column and get the values
    author_values = author_sample.drop(["Train/Test", "Author"], axis=1).values

    #Creating a tensor with the author row
    author_tensor = torch.FloatTensor(author_values)
    return author_tensor
    
def get_sample(data, size):
    tensors_labels = []
    tensors = []
    labels = []
    for i in range(size):
        same_author = random.choice([0, 1]) # 0 = not from same author, 1 same author
        author1, author2 = get_random_pair_of_author(data, same_author)

        author1_tensor = get_random_tensor(data, author1)
        author2_tensor = get_random_tensor(data, author2)

        #We use autograd.Variable to help us handling the two tensors
        tensors.extend(autograd.Variable(torch.FloatTensor((author1_tensor + author2_tensor))))
        labels.extend(torch.FloatTensor([same_author]))

    tensors_labels.append(tensors)
    tensors_labels.append(labels)
    
    #Return a list of two list: a list of tensors and a list of labels
    return tensors_labels

def read_csv_file(filepath):
    #Reading the file and getting a dataframe from the csv file
    df = pd.read_csv(filepath)

    #Getting all the data labeled as Train
    train_df = df[df["Train/Test"] == "Train"]
    train_df.reset_index(inplace=True, drop=True)
    
    #Getting all the data labeled as Test
    test_df = df[df["Train/Test"] == "Test"]
    test_df.reset_index(inplace=True, drop=True)

    return df, train_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("--n_epochs", dest="n_epochs", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10, help="The size of the batch for each iteration")

    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=0, help="The size of the hidden layer.")
    parser.add_argument("--nonlin", dest="nonlin", type=str, default=None, choices=['ReLU', 'Tanh'], help="Non Linear function for part3.Options are  \"ReLU\" or \"Tanh\".")
    parser.add_argument("--train_size",  dest="train_size", type=int, default=200, help="Number of pair of authors to train with.")
    parser.add_argument("--test_size",  dest="test_size", type=int, default=20, help="Number of pair of authors to train with.")

    parser.add_argument("--learning_rate", dest="learning_rate", type=int, default=0.01, help="learning ratio used by the network's optim algorithm")
        
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    data, train_data, test_data = read_csv_file(args.featurefile)

    # implement everything you need here
    classifier = AuthorPerceptron(input_size=(data.shape[1]-2), hidden_size= args.hidden_size, activation_fn=args.nonlin)

    #// operand ensures the division will give an integer ( the floor operation)
    classifier.train(train_data, args.batch_size, args.n_epochs, (args.train_size//args.batch_size), args.learning_rate)
    classifier.test(test_data, args.test_size)
