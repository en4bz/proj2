from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

labels = ['cs','math','physics','stat']

P = []
A = []

for line in open('confusion.txt'):
    sp = line.strip().split(',')
    A.append(sp[0])
    P.append(sp[1])

cm = confusion_matrix(A, P)

# Show confusion matrix in a separate window
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, interpolation='nearest')
fig.colorbar(cax)
plt.ylabel('True label',fontsize=24)
plt.xlabel('Predicted label',fontsize=24)


plt.tick_params(axis='both', which='major', labelsize=18)
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)

plt.show()
