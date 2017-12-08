import numpy as np
from math import log

def vector_entropy(vector, base = -1):
    '''
    Computes entropy of a vector.

    Given vector and an optional base, normalizes the
    vector using the 1-norm and computes the entropy using
    the well known formula. Base defaults to the length of
    the input vector, to normalize across lengths of vectors
    if needed.
    '''
    vector=np.array(vector)
    if base <= 0:
        base=len(vector)

    # normalize
    s = float(sum(vector))
    vector = vector/s

    # calculate result
    result = sum(-x*entropy_log(x,base) for x in vector)
    return result

def entropy_log(x, base):
    '''
    Returns the logarithm of x in given base, returns 0 when x is 0.
    '''
    if x==0:
        return 0
    return log(x, base)

if __name__ == '__main__':
    print('Almost 1:')
    print(vector_entropy(np.array(np.random.uniform(size=1000))))
    print(vector_entropy(np.array(np.random.uniform(size=100000))))
    print('\nRoughly increasing by 1 at every step:')
    for b in range(3, 19):
        print(vector_entropy(np.array(np.random.uniform(size=2**b)), 2))
    print('\nUniform vector (x_i=x_j for all i,j): ')
    print(vector_entropy(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])))
    print('Uniformly distributed vector:')
    print(vector_entropy(np.array([1,2,3,4,5,6,7,8,9,0])))
    print('Almost 1-hot vector:')
    print(vector_entropy(np.array([1,2,2,4,4,5,3,5,4,3,1880,3,1])))
    print('1-hot vector:')
    print(vector_entropy(np.array([0,0,0,1,0,0,0,0,0,0])))
