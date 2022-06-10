#!/usr/bin/env python3
# --------------------------------------------------------------
# Post-processing for MintPy output results
#
# Yuan-Kai Liu, 2022-2-10
# --------------------------------------------------------------

# Recommended usage:
#   import sarut.tools.math as sarmath

import numpy as np
import matplotlib.pyplot as plt


#############################################################

def rect_pulse(X):
    # Define a rectangular function
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


def chirp_generate(Amp=1, w0=5, Br=4, tau=10):
    print('This is a chirp waveform example.')

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
    return x, main, rect


def disp_arctan(X, s, d):
    """
    Define an interseimic arctan deformation function at the surface
    A deformation function taken from Paul Segall's book:
    EARTHQUAKE AND VOLCANO DEFORMATION, 2010, page 40, eq(2.32).
    X   distance array
    s   slip rate
    d   locking depth
    """
    u = np.zeros(X.shape)
    i = 0
    for x1 in X:
        u[i] = (s/np.pi) * np.arctan(x1/d)
        i = i + 1
    return u


def gridsearch_disp(d=1, s=5):
    """
    Given a arctan interseimic model (s: slip rate; d: locking depth)
    Add Gaussian noise
    Find parameters grid search (testing...)
    Plot it
    """
    x = np.linspace(-10, 10, 201)
    u = disp_arctan(x,s,d)

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

    # Try in a grid-search way to find parameters
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

    print('Least-squares error: ', ERR)
    print('Estimated s = ', sz[ms])
    print('Estimated d = ', dz[ns])

    uz = disp_arctan(x,sz[ms],dz[ns])

    # Scatter plot of the noisy data points & estimated curve
    fig2 = plt.figure()
    plt.scatter(x, u_obs, marker='o', s=5)
    plt.plot(x, uz)
    plt.xlabel('Distance from fault (x)'); plt.ylabel('Displacement (u)'); plt.axis('tight'); plt.title('Interseismic deformation')
    plt.show()



def networking(friends, sortbylen=True):
    # person's friendship circle (network) is a person themselves
    # plus friendship circles of all their direct friends
    # minus already seen people
    # https://stackoverflow.com/questions/15331877/creating-dictionaries-of-friends-that-know-other-friends-in-python
    def friends_graph(people_all, friends):
        # graph of friends (adjacency lists representation)
        G = {p: [] for p in people_all} # person -> direct friends list
        for friend in friends:
            for p in friend:
                f_list = list(friend)
                f_list.remove(p)
                G[p].extend(f_list)
        return G

    def friendship_circle(person): # a.k.a. connected component
        seen.add(person)
        yield person
        for friend in direct_friends[person]:
            if friend not in seen:
                yield from friendship_circle(friend)

    people_all = list(set(sum(friends, [])))
    direct_friends = friends_graph(people_all, friends)
    seen = set() # already seen people

    # group people into friendship circles
    circs = (friendship_circle(p) for p in people_all if p not in seen)

    # convert generator to a list with sublist(s)
    circ_lists = []
    for circ in circs:
        member = sorted(list(circ))
        circ_lists.append(member)

    if sortbylen:
        circ_lists.sort(key=len, reverse=True)

    return circ_lists