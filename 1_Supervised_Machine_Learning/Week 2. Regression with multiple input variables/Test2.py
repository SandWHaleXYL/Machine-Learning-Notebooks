import time
import copy, math
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2) 

a=np.zeros(4)
#Cost Function J(w,b) with multiple variables
def compute_cost(X,y,W,b):
    m=X.shape[0]
    cost_sum=0
    for i in range(m):
        f_wb=np.dot(X[i],W)+b
        cost=(f_wb-y[i])**2
        cost_sum+=cost
    total_cost=cost_sum/(2*m)
    return total_cost

#Calculating each gradient step
def compute_gradient(X,y,W,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        f_wb=np.dot(X[i],W)+b
        error=f_wb - y[i]
        for j in range(n):
            dj_dw[j]+=error*X[i,j]
        dj_db+=error
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

#Gradient Descent
def gradient_descent(X,y,W_in,b_in,alpha,num_iters):
    #w=W_in
    W=copy.deepcopy(W_in) #avoid modifying global w within function
    b=b_in
    J_history=[]

    for i in range(num_iters):
        dj_dw,dj_db=compute_gradient(X,y,W,b)
        W=W - alpha*dj_dw
        b=b - alpha*dj_db

        if i<100000:
            cost=compute_cost(X,y,W,b)
            J_history.append(cost)

        if i% math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
            #print(f"Iteration {i:4d}: Cost {float(cost):8.2f}   dj_dw: {dj_dw:8.2f}   dj_db: {dj_db:8.2f}  w: {W:0.2f}  b:{b:0.2f}")
    return W,b,J_history

# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])
# # W_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
# # b_init = 785.1811367994083
# # final_cost=compute_cost(X_train, y_train, W_init, b_init)
# # print(f"Final cost: {final_cost}")
# W_init=np.zeros(X_train.shape[1])
# b_init=0.0
# iterations=10000
# alpha=5.0e-7
# w_final,b_final,J_hist=gradient_descent(X_train, y_train, W_init, b_init, alpha, iterations)

X_train, y_train=load_house_data()
X_features=['size(sqft)','bedrooms','floors','age']
W_init=np.zeros(X_train.shape[1])
b_init=0.0
iterations=10000
alpha=5.0e-7

# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()
w_final,b_final,J_hist=gradient_descent(X_train, y_train, W_init, b_init, alpha, iterations)