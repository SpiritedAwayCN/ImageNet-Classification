import numpy as np
from numpy import linalg as la

EX = [[102.38590308, 114.03596477, 120.20542635]]
EXY = [[15875.05715902, 16286.31198753, 16212.65198293],
       [16286.31198753, 17917.16620909, 18175.70219696],
       [16212.65198293, 18175.70219696, 19657.62259017]]
a = EXY - np.dot(np.array(EX).T, np.array(EX))
print(a)
u, sigma, vt = la.svd(a)
print(sigma)
print(vt)
