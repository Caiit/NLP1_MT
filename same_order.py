import numpy as np

# naming is work in progress....
def same_order(attention_matrix):

    #to be sure
    attention_matrix = np.array(attention_matrix)

    grid = np.zeros(attention_matrix.shape)
    grid[0,:] = attention_matrix[0,:]
    for i in range(1, grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i,j] = attention_matrix[i,j] + max(grid[i-1, :j+1])
    return max(grid[-1,:]) / grid.shape[0]

if __name__ == '__main__':
    print('high:', same_order([[1,0,0], [0,1,0], [0,1,0], [0,0,1]]))
    print('low:', same_order([[0,0,1], [0,1,0], [0,1,0], [1,0,0]]))
