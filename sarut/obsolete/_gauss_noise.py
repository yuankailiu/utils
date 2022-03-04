# This is a only practice of using radom function in python to create Gaussian noise data points

import numpy as np
import matplotlib.pyplot as plt 


# Parameters definition: Mean value (mu) & Standard deviation (sd)
mu = 0
sd = 0.1

# Call the function
noise = np.random.normal(mu, sd, 1000)

# abs(mu - np.mean(noise)) < 0.01
# abs(sd - np.std(noise, ddof=1)) < 0.01
print('Mean:',np.mean(noise))
print('Standard deviation:',np.std(noise))

fig1 = plt.figure()
count, bins, ignored = plt.hist(noise, 100, density=True)
plt.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sd**2) ), linewidth=2, color='r')
plt.show()

x = np.arange(1,1001,1)
fig2 = plt.figure()
plt.scatter(x, noise, marker='o', s=5)
plt.show()

# fig1 = plt.figure()
# plt.plot(x, u)
# plt.xlabel('Distance from fault (x)'); plt.ylabel('Displacement (u)'); plt.axis('tight'); plt.title('Interseismic deformation')
# plt.show()
