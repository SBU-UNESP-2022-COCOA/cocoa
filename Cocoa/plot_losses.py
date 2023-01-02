import numpy as np
import matplotlib.pyplot as plt

losses_train = np.loadtxt('losses.txt')[0]
losses_test  = np.loadtxt('losses.txt')[1]

#epoch = np.arange(len(losses_test)) + 1
epoch = np.arange(len(losses_test)) + 1

plt.plot(epoch, losses_train, 'o-', label="loss_train")
plt.plot(epoch, losses_test,  'o-',  label="loss_validation")

# plt.ylim([0,10])
plt.yscale('log')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig("losses.pdf")