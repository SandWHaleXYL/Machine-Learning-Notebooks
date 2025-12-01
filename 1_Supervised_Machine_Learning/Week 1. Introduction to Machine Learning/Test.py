import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition,plt_stationary,plt_update_onclick,soup_bowl,plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0,2.0,3.0,4.0,5.0])
y_train = np.array([300.0,500.0,700.0,900.0,1100.0])

#Cost Function
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum+=cost
    total_cost=cost_sum/(2*m)
    return total_cost

#plt_intuition(x_train,y_train)

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)