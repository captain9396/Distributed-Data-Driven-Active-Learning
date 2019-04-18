import numpy as np

n = 100
m = 1000
a = np.random.randint(0,255, (n,n))
fp = open("sqr.txt", "w")
s = ""
for i in range(n):
    for j in range(n):
        s += str(i) + " " +str(j) + " " +str(a[i][j]) + '\n'


fp.write(s)
