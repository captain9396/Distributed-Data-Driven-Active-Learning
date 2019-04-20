import numpy as np
n = 50000
m = 1000
a = np.random.randint(0,255, (n,m))

np.savetxt('vectors_'+str(n)+'x'+str(m)+'.txt', a)
