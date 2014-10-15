from matplotlib import pyplot as plt
import sys

with open(sys.argv[1]) as f:
    plt_points = []
    for line in f:
        p = line.split(" ")
        plt_points.append((int(p[0]), float(p[1]))
        
        )

plt_points = sorted(plt_points, key=lambda d: d[0])
x, y = zip(*plt_points)
plt.plot(x, y, 'g-', x, y, 'go')
plt.xlabel('Number of features', fontsize=15)
plt.ylabel('Success Rate', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.legend(loc="lower right",fontsize=24)
plt.grid()
plt.show()
