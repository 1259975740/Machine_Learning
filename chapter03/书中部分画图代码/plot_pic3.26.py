import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
n_samples = 300
n_features = 2
X1 = np.random.rand(n_samples, n_features)
y1 = np.ones((n_samples, 1))
idx_neg = (X1[:, 0] - 0.5) ** 2 + (X1[:, 1] - 0.5) ** 2 < 0.1
y1[idx_neg] = 0
blue = X1[y1.ravel()==0]
red = X1[y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=50,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='+',s=80,label='class 2')
plt.legend(loc='best')



y1 = np.ones((n_samples, 1))
idx_neg = (X1[:, 0] < 0.5) * (X1[:, 1] < 0.5) + (X1[:, 0] > 0.5) * (X1[:, 1] > 0.5)
y1[idx_neg] = 0
plt.figure()
blue = X1[y1.ravel()==0]
red = X1[y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=50,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='+',s=80,label='class 2')
plt.legend(loc='best')

rho_pos = np.random.rand(n_samples // 2, 1) / 2.0 + 0.5
rho_neg = np.random.rand(n_samples // 2, 1) / 4.0
rho = np.vstack((rho_pos, rho_neg))
phi_pos = np.pi * 0.75 + np.random.rand(n_samples // 2, 1) * np.pi * 0.5
phi_neg = np.random.rand(n_samples // 2, 1) * 2 * np.pi
phi = np.vstack((phi_pos, phi_neg))
X1 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y1 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
plt.figure()
blue = X1[y1.ravel()==0]
red = X1[y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=50,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='+',s=80,label='class 2')
plt.legend(loc='best')

rho_pos = np.random.rand(n_samples // 2, 1) / 2.0 + 0.5
rho_neg = np.random.rand(n_samples // 2, 1) / 3.5
rho = np.vstack((rho_pos, rho_neg))
phi_pos = np.pi * 1+ np.random.rand(n_samples // 2, 1) * np.pi 
phi_neg = np.random.rand(n_samples // 2, 1) * 2 * np.pi*2.2
phi = np.vstack((phi_pos, phi_neg))
X1 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y1 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
plt.figure()
blue = X1[y1.ravel()==0]
red = X1[y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=50,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='+',s=80,label='class 2')
plt.legend(loc='best')
