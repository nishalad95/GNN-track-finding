import numpy as np
import matplotlib.pyplot as plt


step1 = np.loadtxt('step1.txt', unpack='False')
step2 = np.loadtxt('step2.txt', unpack='False')

# set bins' interval for your data
# You have following intervals: 
# 1st col is number of data elements in [0,10000);
# 2nd col is number of data elements in [10000, 20000); 
# ...
# last col is number of data elements in [100000, 200000]; 

plt.hist(step1, histtype='bar', bins=50)
plt.xlabel('Extrapolation Step Size s')
plt.ylabel('Frequency')
plt.title('step1: -x\'/Vx\'')
plt.legend()
plt.show()

plt.hist(step2, histtype='bar', bins=500)
plt.xlim([-0.5e+8, 0.5e+8])
plt.xlim([-2e+7, 2e+7])
plt.xlabel('Extrapolation Step Size s')
plt.ylabel('Frequency')
plt.title('step2: -2Vx\'/Ax\' + x\'/Vx\'')
plt.legend()
plt.show()