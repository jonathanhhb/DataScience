#!/usr/bin/python
"""
This code was mostly taken from an onlinle demo, robably TowardDataScience.com or MachineLearningMastery.com.
The idea is to demo Random Forests in python using SKLearn.
I adapted this to Brittany's Polio Surveillance Cost dataset and got much faster results than the NN approach,
and interestingly some different results.
TODO: Make a version of this that "just works" on any dataset where there are a bunch of input columns anda an output col.
"""
import sys
import pdb

# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


# Read in data and display first 5 rows
features = pd.read_csv( sys.argv[1] )
features.head(5)

# Descriptive statistics for each column
features.describe()

do_one_hot = True
if do_one_hot:
# One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
    features.iloc[:,5:].head(5)

output_label = features.columns[-1]

# Labels are the values we want to predict
labels = np.array(features[ output_label ])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop( output_label, axis = 1)
#features= features.drop( 'UID', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
print( feature_list )
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 420)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

do_baseline = False
if do_baseline:
# The baseline predictions are the historical averages
    baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
abs_errors = abs(predictions - test_labels)
errors = abs((predictions - test_labels)/test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(abs_errors), 2), 'degrees.')
print('Mean % Error:', round(np.mean(errors), 2), 'degrees.')

#importances = sorted(list(rf.feature_importances_))
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
#print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances
print( str( feature_importances ) )

# Import matplotlib for plotting 
import matplotlib.pyplot as plt
# .. and use magic command for Jupyter Notebooks
# %matplotlib inline

# Set the style
#plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))
x_labels = [ elem[0] for elem in feature_importances ]
# Make a bar chart
plt.bar(x_values, sorted(importances, reverse=True), orientation = 'vertical')
# Tick labels for x axis
plt.xticks(np.arange(len(x_values)), x_labels, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()
