# Neural Network Charity Analysis

## Overview 

For our 19th data analytics project, we used deep-learning neural networks with TensorFlow, to analyze the success of charitable donations.  We pre-processed the data to normalize the values and turn the text values into numbers that the machine learning algorithms can understand.  Then we trained, compiled, and evaluated the model.  We finished off the project with an attempt to optimize the model further by changing some of the process settings.  

___

## Resources 
- Data Source: [charity_data](Resources/charity_data.csv) 
- Software: Python 3.7.13; Jupyter Notebook, TensorFlow, sklearn, pandas

# Results
## Preprocessing 
- I removed the columns "EIN" and "NAME" since they had no baring on the results of the donations.
- For APPLICATION_TYPE and "CLASSIFICATION", I reduced the number of items in each column through binning, where I created an "Other" item for values with low counts.  
- IS_SUCCESSFUL is the column that contains the binary data indicating if the donation was successful, as such it was considered our target variable for the deep learning network.
- APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the remaining features for the model.  
- Used OneHotEncoder to encode the categorical variables so that all the features are numerical.
- Used StandardScaler to scale the data into a more normal form.

## Train, Compile, and Evaluate the Model
- Use used the keras Sequential Model for our Deep Learning Neural Network
- We fit the model with 3 layers the first layer was given 80 neural nodes and the second layer was given 30 neural nodes.  The input dimensions for the first matched the number of features (44).  
- The last layer was our sigmoid output layer.  
- For the compilation, the optimizer is adam and the loss function is binary_crossentropy.
- The model accuracy was 72.5%. Which did not satisfy the 75% requirement to predict the outcomes of the success of donations.  
- To try and increase the performance of the model I attempted a few changes.  
    - I changed the activation to the tanh and increased the number of nodes for the two hidden layers to 100 and 50.
    - I also tried changing the optimizer and loss to sgd and mse.
    - Third, I tried adding 2 additional hidden layers, with nodes (100, 60, 40,20) for the four layers.
    - Lastly I went back and took the log value for the ASK_AMT to see if I could reduce variance and have a more normal distribution.  

# Summary
My deep learning model did not reach the 75% accuracy goal. It did vary a little, but not much from the 72.5%.  Additionally I would suggest trying to use a supervised Machine Learning model, possibly Random Forest Classifier and compare the performance against this deep learning model.  


