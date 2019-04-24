import numpy as np

def get_xor_data(n,d):
    n_samples = int(n/2)
    dimension = d
    a = [[np.random.uniform(0,1) for d in range(dimension)] + [0] for i in range(0,n_samples)] + [[np.random.uniform(1,2) for d in range(dimension)] + [0] for i in range(0,n_samples)]
    b = [[np.random.uniform(0,1),np.random.uniform(1,2)] + [np.random.uniform(0,1) for d in range(dimension-2)] + [1] for i in range(0,n_samples)] + [[np.random.uniform(1,2),np.random.uniform(0,1)] + [np.random.uniform(0,1) for d in range(dimension-2)] + [1] for i in range(0,n_samples)]
    return np.array(a + b)












N = 100000
D = 100
np.savetxt('xor.txt', get_xor_data(N, D))
