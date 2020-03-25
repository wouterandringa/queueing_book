import numpy as np
import matplotlib.pylab as plt
import tikzplotlib
from matplotlib import style
style.use('ggplot')

np.random.seed(3)
a = np.random.poisson(11.8, 1000)


def computeQ(a, C, Q0=0):  # initial queue length is 0
    N = len(a)
    Q = np.empty(N)  # make a list to store the values of  Q
    d = np.empty(N)  # make a list to store the values of  d
    Q[0] = Q0
    for n in range(1, N):
        d[n] = min(Q[n-1] + a[n], C[n])
        Q[n] = Q[n-1] + a[n] - d[n]
    return Q


def unbalanced():
    p = np.empty([5, len(a)])
    p[0, :] = 1. * np.ones_like(a)
    p[1, :] = 1. * np.ones_like(a)
    p[2, :] = 1. * np.ones_like(a)
    p[3, :] = 3. * np.ones_like(a)
    p[4, :] = 9. * np.ones_like(a)
    return p

p = unbalanced()

# holidays.
for j in range(len(a)):
    Id = j % 5
    p[Id, j] = 0

s = np.sum(p, axis = 0)

Qub= computeQ(a,s)

def balanced():
    p = np.empty([5,len(a)])
    p[0,:] = 2.*np.ones_like(a)
    p[1,:] = 2.*np.ones_like(a)
    p[2,:] = 3.*np.ones_like(a)
    p[3,:] = 4.*np.ones_like(a)
    p[4,:] = 4.*np.ones_like(a)
    return p

p = balanced()

for j in range(len(a)):
    Id = j%5
    p[Id,j] = 0

s = np.sum(p,axis = 0)
Qbb= computeQ(a,s)

p = unbalanced()
for j in range(int(len(a)/5)):
     p[:,5*j] = 0
s = np.sum(p,axis = 0)
Quu= computeQ(a,s)

p = balanced()
for j in range(int(len(a)/5)):
     p[:,5*j] = 0
s = np.sum(p,axis = 0)
Qbu= computeQ(a,s)

mins = np.minimum.reduce([Qbb, Qub, Qbu, Quu])
maxs = np.maximum.reduce([Qbb, Qub, Qbu, Quu])


plt.figure(figsize=(10, 2))
plt.plot(mins, label="min")
plt.plot(maxs, label="max")
# plt.plot(Qbb, label="bal cap, spread out")
# plt.plot(Qub, label="unbal capacity, simul")
# plt.plot(Qbu, label="bal cap, spread out")
# plt.plot(Quu, label="unbal capacity, simul")
plt.xlabel("time (weeks)")
plt.ylabel("Q")
plt.legend(loc="upper left")# , bbox_to_anchor=(1,1))
tikzplotlib.save('balanced.tex') #, figureheight='6cm', figurewidth='15cm')
plt.close() 

# now we control the service rate. If the queue is longer than some
# threshold we let the server do some extra work, while if the queue
# is smaller than some other threshold, the server has to do less.

def computeQExtra(a, s, e, Q0=0): #  initial queue length is 0
    N = len(a)
    Q = [0]*N # make a list to store the values of  Q
    Q[0] = Q0
    for n in range(1,N):
        q = Q[n-1]
        if  q <12:
            s[n] = max(s[n]-e,0) # protect from becoming negative
        elif q >= 24:
            s[n] += e
        Q[n] = max( q +a[n] - s[n], 0)
    return Q

np.random.seed(3)
a = np.random.poisson(11.8, 1000)
s = 12.*np.ones_like(a)
Q= computeQ(a,s)
Qe1= computeQExtra(a,s,1)
Qe2= computeQExtra(a,s,2)
Qe5= computeQExtra(a,s,5)

plt.figure(figsize=(6, 2))
plt.ylim(0, 200)
plt.plot(Q, label="Q", color='black')
plt.plot(Qe1, label="Qe1", color='green')
plt.plot(Qe2, label="Qe2", color='blue')
plt.plot(Qe5, label="Qe5", color='red')
plt.xlabel("time (weeks)")
plt.ylabel("Q")
plt.legend()
# plt.legend(loc="upper left")# , bbox_to_anchor=(1,1))
tikzplotlib.save('service_control.tex') #, figureheight='6cm', figurewidth='15cm')
plt.close() 
