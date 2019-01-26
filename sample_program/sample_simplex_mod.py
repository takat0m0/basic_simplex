# -*- coding:utf-8 :-*-

import os
import sys
import numpy as np

def get_positive_y_max(b_bar, y):
    assert(len(b_bar) == len(y))

    j_min_ = None
    theta = float('inf')
    
    for i, (b_, y_) in enumerate(zip(b_bar, y)):
        if y_ <= 0.0:
            continue
        tmp = b_/y_
        if theta > tmp:
            theta = tmp
            j_min_ = i
    return j_min_, theta



if __name__ == '__main__':
    A = np.asarray([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0, 1.0, 0.0],		    
                    [2.0, -1.0, 0.0, 0.0, 0.0, -1.0]])
		    
    
    b = np.asarray([2.0, 3.0, 1.0, -1.0])
    c = np.asarray([-1.0, -2.0, 0.0, 0.0, 0.0, 0.0]) # need 0.0
    
    # initial index (they must be feasible...)
    B_index = np.asarray([2, 3, 4, 5])
    N_index = np.asarray([0, 1])

    # initial feasible solution
    B = A[:, B_index]
    B_inverse = np.linalg.inv(B)
    b_bar = np.dot(B_inverse, b)
    x = np.zeros(len(c), dtype = float)
    for x_, idx in zip(b_bar, B_index):
        x[idx] = x_

    # simplex loop
    while(True):
        c_B = c[B_index]
        c_N = c[N_index]
        B = A[:, B_index]
        N = A[:, N_index]
        B_inverse = np.linalg.inv(B)

        # pricing
        tmp = np.dot(B_inverse.transpose(), c_B)
        rho = c_N - np.dot(N.transpose(), tmp)
        k_ = np.argmin(rho)
        k = N_index[k_]

        if rho[k_] >= 0.0:
            break
    
        # ratio test
        b_bar = np.dot(B_inverse, b)
        y = np.dot(B_inverse, A[:, k])
        j_min_, theta = get_positive_y_max(b_bar, y)
        j_min = B_index[j_min_]
        
        # pivot
        x[k] = theta
        for tmp, idx in zip(y, B_index):
            x[idx] -= tmp * theta
        print(x)
        B_index = np.insert(B_index[B_index != j_min], 0, k)
        N_index = np.insert(N_index[N_index != k], 0, j_min)    
        
    print(x)
    print('obj = {}'.format(np.dot(x, c)))
