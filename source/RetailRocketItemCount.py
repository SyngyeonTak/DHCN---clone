# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:04:44 2024

@author: fge50
"""

import pickle


train_data = pickle.load(open('../datasets/RetailRocket/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/RetailRocket/test.txt', 'rb'))

all_data_X = train_data[0]+test_data[0]
all_data_Y = train_data[1]+test_data[1]


print(all_data_Y)


flat_list_X = [item for sublist in all_data_X for item in sublist]

# Count unique item values
unique_items = len(set(flat_list_X + all_data_Y))

print("Number of unique item values:", unique_items)