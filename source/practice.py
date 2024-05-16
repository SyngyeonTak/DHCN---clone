import numpy as np
import torch
import math
from scipy.sparse import csr_matrix

# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:57:55 2024

@author: fge50
"""


class MyModel(torch.nn.Module):
    def __init__(self, emb_size):
        super(MyModel, self).__init__()
        self.emb_size = emb_size
        self.fc = torch.nn.Linear(emb_size, emb_size)

    def forward(self, x):
        return self.fc(x)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

# Example usage
model = MyModel(100)  # Initialize model with embedding size 5
model.init_parameters()  # Initialize parameters
for param in model.parameters():
    print(param)
    

tensor = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
result = tensor.unsqueeze(0)

print(result)    


# def data_masks(all_sessions, n_node):
#     indptr, indices, data = [], [], []
#     indptr.append(0)
    
#     # print("Iteration 0:")
#     # print("indptr:", indptr)
#     # print("indices:", indices)
#     # print("data:", data)
    
#     for session in all_sessions:
#         unique_session = np.unique(session)
#         length = len(unique_session)
#         s = indptr[-1]
#         indptr.append((s + length))
        
#         # print("\nSession:", session)
#         # print("unique_session:", unique_session)
#         # print("Iteration", len(indptr) - 1, ":")
        
#         for item in unique_session:
#             indices.append(item - 1)  # Shifting to 0-based indexing
#             data.append(1)

     
#     print("indptr:", indptr)
#     print("indices:", indices)
#     print("data:", data)   
#     matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
#     return matrix

# # Session data
# session_1 = [1, 3, 3, 4, 5]
# session_2 = [3, 4, 6, 7]
# session_3 = [5, 4, 7, 11, 5]

# all_sessions = [session_1, session_2, session_3]
# n_node = 11  # Assuming the largest item index is 11

# # Generating the sparse matrix representation
# sparse_matrix = data_masks(all_sessions, n_node)

# # Output the sparse matrix
# print("\H_T Matrix Representation:")
# print(sparse_matrix.T.toarray())
# print("\n(H_T.sum(axis=1).reshape(1, -1))")
# print(sparse_matrix.sum(axis=1).reshape(1, -1))
# print("\n(1.0/H_T.sum(axis=1).reshape(1, -1))")
# print(1.0/sparse_matrix.sum(axis=1).reshape(1, -1))
# print("\n(1.0/H_T.sum(axis=1).reshape(1, -1))")
# print(sparse_matrix.T.multiply(1.0/sparse_matrix.sum(axis=1).reshape(1, -1)))
# print(sparse_matrix.T.multiply(sparse_matrix.sum(axis=1).reshape(1, -1)))
