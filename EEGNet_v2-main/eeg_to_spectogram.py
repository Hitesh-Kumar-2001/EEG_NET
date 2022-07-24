import numpy as np
from scipy import signal
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests  # pip install requests
mpl.rcParams['figure.figsize'] = [12, 8]
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from scipy.signal import spectrogram

def read_csv(path):
    sample = np.genfromtxt(path,delimiter=',')
    sample2 = []
    for i in sample:
         temp = i[np.logical_not(np.isnan(i))]
         sample2.append(temp)
    return np.array(sample2)

a = read_csv("D:\SSVEP-Neural-Generative-Models-master/bci_iv_2a_data_preprocessed\A01/test/0/1_0.csv")
nfft = 1124
fs = 250
def spectogram(data):
    size = data.shape[0]
    imdata = []
    spegram_data = []
    print(size)
    for i in range(size):
        print(i)
        im = specgram(data[i], NFFT=nfft, Fs=fs, noverlap=nfft/2,cmap="rainbow")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.savefig("image.png", bbox_inches = 'tight',pad_inches = 0)
        img = Image.open("image.png")
        imdata.append(np.asarray(img))
        spegram_data.append((im))
    imdata = np.array(imdata)
    return imdata, spegram_data

if __name__ == "__main__":
    imdata ,_= spectogram(a)
    print(imdata)
