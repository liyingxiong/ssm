'''
Created on Feb 27, 2015

@author: Li Yingxiong
'''
import numpy as np
from scipy.sparse import csr_matrix

a = np.zeros((4, 1))
b = np.ones((4, 1))

print np.dot(a.T, b)
