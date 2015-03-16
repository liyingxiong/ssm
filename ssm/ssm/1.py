'''
Created on Feb 25, 2015

@author: Li Yingxiong
'''

import numpy as np
from scipy.sparse import coo_matrix

a = np.array([[0, 0, 0, 1, 0, 1, 0],
              [1, 0, 1, 0.5, 0, 0, 0]])

b = np.array([[0, 3, 6, 1, 0, 1, 0],
              [1, 0, 1, 0.5, 8, 4, 0]])

k = np.array([[1., -1.],
              [-1., 1.]])

print a.T.dot(k).dot(b)
print b.T.dot(k).dot(a)

# a = np.array([[0.2457,  0.2457,  0.,  0.,  0.2543,  0.2543,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#               [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.2457,  0.2457,  0.,  0.,  0.2543,  0.2543]])

# l = []
# for i in range(4):
#     a[0, 0] = i
#     b = coo_matrix(a)
#     print b
#     print np.copy(b)
#     l.append(b)
#
# print l[0]
# print l[1]
# print l[2]

# b = coo_matrix(a)
#
# d = np.copy(a)
#
# print d


k = np.array([[1., -1.],
              [-1., 1.]]) / (2 * np.sqrt(2))
np.set_printoptions(precision=4, linewidth=1000)

print a.T.dot(k).dot(a)


# print np.nonzero(a)[1]
# print np.unique(np.nonzero(a)[1])
#
# idx = np.unique(np.nonzero(a)[1])
# print a.T[np.unique(np.nonzero(a)[1])]
#
# b = a.T[np.unique(np.nonzero(a)[1])]
#
# print b.dot(k).dot(b.T)
#
# d = b.dot(k).dot(b.T)
#
# row = np.repeat(idx, len(idx))
# print row
#
# col = np.tile(idx, len(idx))
# print col
#
# print coo_matrix((d.flatten(), (row, col)), shape=(7, 7)).todense()
