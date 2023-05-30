#!/usr/bin/env python
# ============================================================================== #
# Plot results
# Powered by xiaolis@outlook.com 202305
# ============================================================================== #
import matplotlib.pyplot as plt
import matplotlib

font = {'family':'Times New Roman', 'size':9}
matplotlib.rc('font', **font)

def read_file(file_name):
    with open(file_name,'r') as f: res = [[float(r) for r in line.split(',')] for line in f.readlines()]
    return res

def plt_one(yvalues, label, color):
    xvalues = range(1, len(yvalues)+1)
    plt.plot(xvalues, yvalues, label=label, color=color, linewidth='0.6')

def plts(ys,labels):
    dtsz = len(labels)
    cmap = plt.get_cmap('brg')
    colors = [cmap(int((i*200)/(dtsz*2))) for i in range(1,dtsz+1)][::-1]
    [plt_one(ys[i],labels[i],colors[i]) for i in range(len(ys))]

def plot_metrics(losses, accs, labs, filename):
    plt.figure(figsize=(10, 4), dpi=150)

    plt.subplot(1, 2, 1)
    plts(losses, labs)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(loc='upper right', ncol=2)

    plt.subplot(1, 2, 2)
    plts(accs, labs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend(loc='lower right', ncol=2)

    plt.savefig(filename,dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    labels = ['- Crop','- ColorJitter','- Grayscale','- Cutout','- AddNoise','- GaussianBlur','- Rotation', '- HorizontalFlip', 'ALL']
    plot_metrics(read_file('loss.sub.txt'), read_file('acc.sub.txt'),labels, 'cifar10.subx8.png')
    labels = ['+ Crop','+ ColorJitter','+ Grayscale','+ Cutout','+ AddNoise','+ GaussianBlur','+ Rotation', '+ HorizontalFlip', 'None']
    plot_metrics(read_file('loss.add.txt'),read_file('acc.add.txt'),labels, 'cifar10.addx8.png')
