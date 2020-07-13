# LT2212 V20 Assignment 3



**PART 1: Creating the feature table**

How to run the program:

	python3 a3_features.py <authors_directory> <output_file> <num_dimensions> [--test test_size]
	python3 a3_features.py enron_sample out.csv 300

The program takes three arguments:
 - the path to the directory that contains the author's directories
 - the name of the output csv file
 - the number of dimensions
 - Optional: the size of the test data. It is a number from 0 and 100. The default value is 20.

Description:

The program a3_features.py generates a csv file. It takes the information from the files located in the path from the first argument. It iterates through the directories and organizes the information by author. To do so, the program follows the next steps:

- The method extract_features receives a list of tokenized texts and makes use of the function TfidfVectorizer
from sklearn to extract the features.
- TruncatedSVD is used to perform dimensionality reduction.
- Finally, a dataframe is created with the reduced feature array. In the method insert_test_train a new
column is inserted with random "Test"/"Train" values. The argument "--test test_size" determines the proportion
of "Test"/"Train" columns.
- The csv is created using the utility function "to_csv" from the dataframe object.


**PART 2: Design and train the basic model**

How to run this part:

    python3 a3_model.py <csv from a3_features.py>
    python3 a3_model.py out.csv

For this part, I'm using the nn.Module class from Pytorch to create a simple Perceptron (AuthorPerceptron) that has a linear layer and a sigmoid. To train the model I'm using SGD algorithm for the optimizer and BCE algorithm for the loss function. The sample data consists on tuples of two rows and a label indicating if the author is the same or not. To randomly select the authors, first we select one of them and then, using the function random.choice, we select a different author depending on a random number (check function get_random_pair_of_author). Then, the method get_:sample returns a list with the authors and the labels. The size of that list corresponds with the batch_size.

As an example of the return value the program generates, this is the result with the default values:
- number of epochs: 3
- batch size: 10
- train size: 100
- learning rate: 0.01

Accuracy: 0.4
Precision: 0.16
Recall: 0.4
F1-measure: 0.2285714285714286


**PART3: Augment the model**

How to run this part:

    python3 a3_model.py <csv from a3_features.py> --nonlin <ReLU/Tanh> --hidden_size <size>
    python3 a3_model.py out.csv --hidden_size 40 --nonlin ReLU

There are several optional arguments:

--n_epochs: Number of epoch. Default value: 3
--batch_size: Size of the batch for each iteration. Default value: 10
--train_size: Size of the training samples. Default: 200
--test_size: Size of the test samples. Default: 20
--learning_rate: Learning rate used by the network's optim algorithm. Default value: 0.01

Description:

For this part, I have added a hidden layer and a non linear layer. For the non linear layer, I'm using Relu and Tanh. The size of the hidden layer is indicated by the parameter hidden_size and the non linear function by the parameter --nonlin which takes the values "ReLU" or "Tanh".

$ python3 a3_model.py out.csv --hidden_size 40 --nonlin ReLU

Accuracy: 0.45
Precision: 0.20249999999999999
Recall: 0.45
F1-measure: 0.2793103448275862


$ python3 a3_model.py out.csv --hidden_size 40 --nonlin Tanh

Accuracy: 0.5
Precision: 0.25
Recall: 0.5
F1-measure: 0.3333333333333333
