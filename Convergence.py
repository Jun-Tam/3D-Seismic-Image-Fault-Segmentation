import numpy as np
import matplotlib.pyplot as plt
import json

name_model1 = './model/weights/model_lr_1.0e-03_25_epochs_hist.txt'
name_model2 = './model/weights/model_lr_1.0e-04_25_epochs_hist.txt'
name_model3 = './model/weights/model_lr_1.0e-05_25_epochs_hist.txt'
hist1 = json.load(open(name_model1, 'r'))
hist2 = json.load(open(name_model2, 'r'))
hist3 = json.load(open(name_model3, 'r'))

def loss_epoch(hist, metric, lr, clr):
    # Accuracy Epoch Plot
    epochs = range(1,len(hist[metric]) + 1)
    label_tr =  'lr: ' + "{:.1e}".format(lr) + ' (train)'
    label_val = 'lr: ' + "{:.1e}".format(lr) + ' (validation)'
    plt.plot(epochs, hist[metric], color = clr, marker = 'o', label=label_tr)
    plt.plot(epochs, hist['val_'+metric], color = clr, marker = 'x', label=label_val)

def plot_metrics(hist1,hist2,hist3,metric):
    loss_epoch(hist1, metric, 1e-3, 'blue')
    loss_epoch(hist2, metric, 1e-4, 'red')
    loss_epoch(hist3, metric, 1e-5, 'green')

#fig, ax = plt.subplots()
fig = plt.figure()
ax1=plt.subplot(1, 2, 1)
plot_metrics(hist1,hist2,hist3,metric='loss')
ax1.set_yticks(np.arange(0,0.05+0.01,0.01))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = 'upper right', fontsize=8)

ax2=plt.subplot(1, 2, 2)
plot_metrics(hist1,hist2,hist3,metric='acc')
ax2.set_yticks(np.arange(0.5,1+0.05,0.1))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right',fontsize=8)
plt.show()
fig.savefig('loss-epoch.png')