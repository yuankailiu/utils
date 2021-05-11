# This is a chirp coding example
print('This is a chirp waveform example.')
import numpy as np
import matplotlib.pyplot as plt 

# Parameters definition
Amp = 1
w0  = 5
Br  = 4
tau = 10


# Define a rectangular function
def rect_pulse(X):
	Y=np.zeros(X.shape)
	i = 0
	for item in X:
		if np.abs(item)<0.5:
			Y[i]=1
		elif np.abs(item)>0.5:
			Y[i]=0
		elif np.abs(item)==0.5:
			Y[i]=0.5
		i = i + 1
	return Y


# x = np.linspace(-2*np.pi, 2*np.pi, 401)
x = np.linspace(-10, 10, 401)
rect = rect_pulse(x/tau)
main = Amp * np.cos( w0*x + Br/(2*tau)*(x**2) )

fig1 = plt.figure()
plt.plot(x, main)
plt.xlabel('Time (s)'); plt.ylabel('Amplitute'); plt.axis('tight')
plt.show()

fig2 = plt.figure()
plt.plot(x, rect)
plt.xlabel('Time (s)'); plt.ylabel('Amplitute'); plt.axis('tight')
plt.show()

fig3 = plt.figure()
plt.plot(x, main*rect)
plt.xlabel('Time (s)'); plt.ylabel('Amplitute'); plt.axis('tight')
plt.show()

