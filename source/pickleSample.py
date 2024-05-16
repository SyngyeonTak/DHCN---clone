# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:16:02 2024

@author: fge50
"""

import pickle

# Assuming train_data and test_data are modified as shown in your code

dataset = ['diginetica', 'Nowplaying', 'RetailRocket', 'Tmall']
slice_int = 500

for i in range(len(dataset)):
    train_data = pickle.load(open('../datasets/' + dataset[i] + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + dataset[i] + '/test.txt', 'rb'))

    # Ensure train_data is a list before slicing
    train_data = list(train_data)  # Convert to list if it's a tuple
    
    # Slice the first 5000 elements for both parts of train_data
    train_data[0] = train_data[0][:slice_int]
    train_data[1] = train_data[1][:slice_int]
    
    # Convert the sliced lists back to tuples
    train_data[0] = tuple(train_data[0])
    train_data[1] = tuple(train_data[1])

    # Ensure train_data is a list before slicing
    test_data = list(test_data)  # Convert to list if it's a tuple
    
    # Slice the first 5000 elements for both parts of train_data
    test_data[0] = test_data[0][:slice_int]
    test_data[1] = test_data[1][:slice_int]
    
    # Convert the sliced lists back to tuples
    test_data[0] = tuple(test_data[0])
    test_data[1] = tuple(test_data[1])
    
    
    # Save train_data
    with open('../datasets/' + dataset[i] + '/train_sliced.txt', 'wb') as train_file:
        pickle.dump(train_data, train_file)
    
    # Save test_data
    with open('../datasets/' + dataset[i] + '/test_sliced.txt', 'wb') as test_file:
        pickle.dump(test_data, test_file)