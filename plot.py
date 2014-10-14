import numpy as np
import matplotlib.pyplot as plt


for S in ['1K','2K','4K','8K','16K']:
    X = []
    Y = []

    for line in open('kNN_' + S + '.txt','r'):
        sp = line.strip().split(',')
        X.append(sp[0])
        Y.append(sp[1])

    plt.plot(X,Y,label=S + ' Features',marker="o")

plt.xlabel('K',fontsize=24)
plt.ylabel('% Correct',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(loc="lower right",fontsize=24)
plt.grid()
plt.show()
