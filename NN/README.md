# ATLAS-UK-ML / NN
ATLAS UK Machine Learning Tutorial - TensorFlow and Neural Networks
-------------------------------------------------------------------------------------------------
This folder contains the TensorFlow-based notebooks that leads from linear regression examples
to build and explore neural networks (NN). The following notebooks exist

  - LinearRegression.ipynb : A linear regression example that allows the user to explore the
    gradient descent algorithm applied to "fitting" y = mx + c to some generated data.

  - FunctionApproximation.ipynb : A single layer neural network example that uses the NN as a function
    approximator to learn the function y = x^2.  This is a familiar function, and illustrates what we
    normally do with a neural network - we take some data that (in general) we don't understand the 
    true model of, and we fit a NN to that data.  The form of the NN, activation functions, nodes
    objective function that we minimise and even the algorithm used to minimise that objective function
    are all chosen by the data scientist.  The output, we hope, a good approximation of the underlying
    true model.  This example allows the user to explore the problem, and to find out how easy or hard
    it is to learn a simple function.  The learing aim of this is for the student to appreciate how
    well the function can be approximated given the right hyperparameters, and how badly the function
    can be approximated.

  - FunctionApproximation2.ipynb : A two layer NN variant of the above.  Again this allows people to 
    explore the application of a neural network to a function approximation problem.  

The aim of these notebooks is to introduce functionality and at the same time to allow the user
to understand the limitations of the methods deployed.  Optimisation is based on a huristic, and 
that has a limited domain of validity; as with function fitting, sometimes the model will converge
to a sensible minimum, sometimes the model will fail to converge completely and sometimes the model
will diverge from a physically meaningful result.  This becomes more apparent when attempting to 
explore more complicated problems; progressing from the linear regression problem to the quadratic
fitting problem.  

Please see the following web page for this material, which includes downloadable tar balls and 
slide deck pdf files:

  https://pprc.qmul.ac.uk/~bevan/teaching/ATLAS-UK-ML.html

-------------------------------------------------------------------------------------------------

If you are interested in machine learning then you may find some of my other machine 
learning tutorial examples online of interest. These can be found at the following page.
  https://pprc.qmul.ac.uk/~bevan/teaching.html

-------------------------------------------------------------------------------------------------
Author: Adrian Bevan (a.j.bevan@qmul.ac.uk)

Copyright (C) Queen Mary University of London

This code is distributed under the terms and conditions of the GNU Public License

-------------------------------------------------------------------------------------------------
