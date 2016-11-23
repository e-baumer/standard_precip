import matplotlib.pyplot as plt
import numpy as np

def plot_index(time, data, save_file=None, index_type='SPI'):
    
    b_width = 22
    pos_inds = np.where(data >= 0.)[0]
    neg_inds = np.where(data < 0.)[0]
    
    data = np.squeeze(data)
    
    fig, ax = plt.subplots()
    ax.bar(
        time[pos_inds], data[pos_inds], width=b_width, align='center', color='b'
    )
    ax.bar(
        time[neg_inds], data[neg_inds], width=b_width, align='center', color='r'
    )
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel(index_type)
    
    if save_file:
        plt.savefig(save_file, dpi=400)
    else:
        plt.show()