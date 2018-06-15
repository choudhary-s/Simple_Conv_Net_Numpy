from scipy import signal
import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-np.tanh(x)**2

def log(x):
    return 1/(1+np.exp(-1*x))
def d_log(x):
    return log(x) * (1-log(x))
#each is a 3 X 3 matrix
x1 = np.array([[0,0,0],[0,0,0],[0,0,0]])
x2 = np.array([[1,1,1],[0,0,0],[0,0,0]])
x3 = np.array([[0,0,0],[1,1,1],[1,1,1]])
x4 = np.array([[1,1,1],[1,1,1],[1,1,1]])

X = [x1,x2,x3,x4]
#print(X[0].shape)
#4 x 1 output
#each output value belongs to separate input
#output of x1 is Y[0], x2 is Y[1], etc.
Y = np.array([[0.53],[0.77],[0.88],[1.1]])
#print(Y[0])

#np.random.seed(5)
#declare weights
w1 = np.random.randn(2,2) * 4 - 1
w2 = np.random.randn(4,1) * 4 - 1

#hyperparameters
epochs = 10000
learning_rate = 0.03

cost_before_train = 0
cost_after_train = 0
final_out, start_out = np.array([[]]), np.array([[]])

for i in range(len(X)):
    l_1 = signal.convolve2d(X[i], w1, 'valid') #2 X 2 matrix as output
    l_1_a = tanh(l_1)
    
    l_1_a_vec = np.expand_dims(np.reshape(l_1_a,-1),axis=0)
    l_2 = l_1_a_vec.dot(w2)
    l_2_a = log(l_2)
    
    #print("size of l_2_a", l_2_a.shape)
    #print("size of l_2", l_2.shape)
    #print("size of l_1_a", l_1_a.shape)
    
    cost = np.square(l_2_a - Y[i]).sum() * 0.5
    cost_before_train += cost
    start_out = np.append(start_out, l_2_a)
print("before training----------------------------------------------------------------------")
print("w1\n", w1,"\nw2\n", w2)
print("cost before train %f" %cost_before_train)
print("start output: ", start_out)

#trianing 
for iter in range(epochs):
    for i in range(len(X)):
        l_1 = signal.convolve2d(X[i], w1, 'valid')
        l_1_a = tanh(l_1)
        
        l_1_a_vec = np.expand_dims(np.reshape(l_1_a, -1), axis=0)
        l_2 = l_1_a_vec.dot(w2)
        l_2_a = log(l_2)
        
        cost = np.square(l_2_a - Y[i]).sum()*0.5
        #print("Current Iter: ", iter," Current train: ", i," Current cost: ", cost)
        
        grad_2_part_1 = (l_2_a - Y[i])
        grad_2_part_2 = d_log(l_2)
        grad_2_part_3 = l_1_a_vec
        grad2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
        
        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_tanh(l_1)
        grad_1_part_3 = X[i]
        
        grad_1_part_1_reshape = np.reshape(grad_1_part_1, (2,2))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        grad1 = np.rot90(signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2), 'valid'), 2)
        
        w2 -= grad2*learning_rate
        w1 -= grad1*learning_rate

for i in range(len(X)):
    l_1 = signal.convolve2d(X[i], w1, 'valid')
    l_1_a = tanh(l_1)
    
    l_1_a_vec = np.expand_dims(np.reshape(l_1_a, -1), axis=0)
    l_2 = l_1_a_vec.dot(w2)
    l_2_a = log(l_2)
    
    cost = np.square(l_2_a - Y[i]).sum()*0.5
    cost_after_train += cost
    final_out = np.append(final_out, l_2_a)
print("After training------------------------------------------------------------------------------")
print("w1\n",w1,"\nw2\n",w2)
print("cost after train: ", cost_after_train)
print("finish output: ", final_out)
print("Actual output: ", Y.T)
        
    
    
    
    