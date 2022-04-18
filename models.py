from cmath import pi
import nn, backend
import math

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        "*** YOUR CODE HERE ***"

        weights = self.get_weights() # Get the weights
        return nn.DotProduct(weights, x) # Return the dot product of the weights and x

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """

        "*** YOUR CODE HERE ***"

        run_x = self.run(x) # Get the dot product of the weights and x
        scalar = nn.as_scalar(run_x)
        if scalar >= 0: # If the dot product is greated than or equal to 0, return 1
            return 1
        else: # Otherwise return 0
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        "*** YOUR CODE HERE ***"

        batch_size = 1 # Batch size
        while True:
            total = 0 # Number of (x, y) in dataset
            total_correct = 0 # Number of (x,y) in dataset that do not need to be updated

            for x, y in dataset.iterate_once(batch_size): # Iterate through dataset

                total+=1 # Add one to total
                predicted_y = self.get_prediction(x) # Get prediction
                y_value = nn.as_scalar(y)

                if predicted_y == y_value:# If the prediction is the same as the real value, add 1 to total_correct
                    total_correct+=1
                else: # Otherwise, update weights
                    self.w.update(y_value, x)
            
            if total == total_correct: # If the weights never needed to be updated, then exit the loop
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here

        "*** YOUR CODE HERE ***"

        self.multiplier = -0.005
        self.hidden = 400
        self.batch = 1

        self.loss_limit = 0.02

        self.w1 = nn.Parameter(self.batch, self.hidden)
        self.b1 = nn.Parameter(1, self.hidden)
        self.w2 = nn.Parameter(self.hidden, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """

        "*** YOUR CODE HERE ***"
        
        # Predict values using f(x) = relu(x*w1 +  b1)*w2 + b2 formula

        xw1 = nn.Linear(x, self.w1)
        predicted_y = nn.AddBias(xw1, self.b1)

        relu = nn.ReLU(predicted_y)

        reluw2 = nn.Linear(relu, self.w2)
        f = nn.AddBias(reluw2, self.b2)

        return f

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***" 

        # Calculate loss

        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """

        "*** YOUR CODE HERE ***"

        while True: # Keep running until loss is less than the limit
            total_loss = 0 # Use this and num_data to calculate average loss across the dataset
            num_data = 0
            for x, y in dataset.iterate_once(self.batch): # iterate through dataset
                num_data+=1

                loss = self.get_loss(x, y) # get loss
                total_loss += nn.as_scalar(loss)

                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients([self.w1, self.w2, self.b1, self.b2], loss)

                self.w1.update(self.multiplier, grad_wrt_w1) # Update values
                self.w2.update(self.multiplier, grad_wrt_w2)
                self.b1.update(self.multiplier, grad_wrt_b1)
                self.b2.update(self.multiplier, grad_wrt_b2)

            if total_loss/num_data < self.loss_limit: # Check if average loss is less than the limit
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.batch = 50 # Trial and error
        self.multiplier = -0.1
        self.hidden = 200

        self.w1 = nn.Parameter(784, self.hidden)
        self.b1 = nn.Parameter(1, self.hidden)
        self.w2 = nn.Parameter(self.hidden, self.hidden)
        self.b2 = nn.Parameter(1, self.hidden)
        self.w3 = nn.Parameter(self.hidden, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # Predict values using f(x) = relu(relu(x * w1 +  b1) * w2 + b2) * w3 + b3 formula

        xw1 = nn.Linear(x, self.w1)
        predicted_y = nn.AddBias(xw1, self.b1)

        relu = nn.ReLU(predicted_y)

        reluw2 = nn.Linear(relu, self.w2)
        predicted_y2 = nn.AddBias(reluw2, self.b2)

        relu2 = nn.ReLU(predicted_y2)

        reluw3 = nn.Linear(relu2, self.w3)
        f = nn.AddBias(reluw3, self.b3)

        return f

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # get loss

        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        while True: # Keep running until loss is less than the limit

            for x, y in dataset.iterate_once(self.batch): # iterate through dataset

                loss = self.get_loss(x, y) # get loss

                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients([self.w1, self.w2, self.b1, self.b2], loss)

                self.w1.update(self.multiplier, grad_wrt_w1) # Update values
                self.w2.update(self.multiplier, grad_wrt_w2)
                self.b1.update(self.multiplier, grad_wrt_b1)
                self.b2.update(self.multiplier, grad_wrt_b2)

            if dataset.get_validation_accuracy() > 0.975: # Check if accuracy is over 97.5 (higher than 97 for buffer room)
                break

