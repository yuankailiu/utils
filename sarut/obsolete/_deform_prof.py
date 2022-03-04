# This is a only practice of how to use Python, numpy sort of things.
# Plot a deformation function taken from Paul Segall's book:
# "EARTHQUAKE AND VOLCANO DEFORMATION", 2010, page 40, eq(2.32).
# Task:
#       1) Plot the function and add Gaussian noise on it, creating a noisy observation.
#       2) Invert the noisy observation and try to find the model parameters.


import numpy as np
import matplotlib.pyplot as plt


# Define a deformation function at the surface
def disp_sur(X,s,d):
	U=np.zeros(X.shape)
	i = 0
	for x1 in X:
		U[i] = (s/np.pi) * np.arctan(x1/d)
		i = i + 1
	return U


# Parameters definition
d = 1
s = 5

# Call the deformation function
x = np.linspace(-10, 10, 201)
u = disp_sur(x,s,d)

# Plot the theoretical deformation profile
fig1 = plt.figure()
plt.plot(x, u)
plt.xlabel('Distance from fault (x)'); plt.ylabel('Displacement (u)'); plt.axis('tight'); plt.title('Interseismic deformation')
plt.show()

# Create Gaussian Noise
mu = 0			# Mean
sd = 0.3		# Standard deviation
noise = np.random.normal(mu, sd, 201)

# Add the noise to the theoretical deformation profile
u_obs = u + noise	# u_obs is the data which contains noise



# Try in a iterative way to find parameters
sz = np.arange(1,20,0.2)
dz = np.arange(1,20,0.2)
ERR  = 10000000
err = np.zeros(x.shape)

m = 0		# index of s
for si in sz:
	n = 0	# index of d
	for di in dz:
		j = 0
		for xi in x:
			err[j] = np.abs(u_obs[j] - (si/np.pi) * np.arctan(xi/di))
			j = j+ 1
		ERRi = np.sum(err)
		if ERRi < ERR:
			ERR = ERRi
			ns  = n
			ms  = m
		else:
			continue
		n = n + 1	# index of d
	m = m + 1		# index of s

print('Least square error: ', ERR)
print('Estimated s = ', sz[ms])
print('Estimated d = ', dz[ns])

uz = disp_sur(x,sz[ms],dz[ns])

# Scatter plot of the noisy data points & estimated curve
fig2 = plt.figure()
plt.scatter(x, u_obs, marker='o', s=5)
plt.plot(x, uz)
plt.xlabel('Distance from fault (x)'); plt.ylabel('Displacement (u)'); plt.axis('tight'); plt.title('Interseismic deformation')
plt.show()
