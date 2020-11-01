'''
Plotting functions
'''

# pylint: disable=C0103, R0912, R0914

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

def bindata(x, y, n_bins=100):
    '''
    Bin 2D data into n bins, e.g. for plotting sigmoid fits
    '''
    data = np.stack((x, y), axis=-1)
    sort_idx = np.argsort(x)
    data = data[sort_idx, :]

    n = data.shape[0]
    binsize = int(np.ceil(n / n_bins))

    binned = np.zeros((n_bins, 2))
    start_idxs = np.arange(0, n, binsize)
    for i, idx in enumerate(start_idxs):
        binned[i, :] = [np.mean(data[idx:idx+binsize, 0]), np.mean(data[idx:idx+binsize, 1])]
    return binned[:, 0], binned[:, 1]

def scatter_cc(x, y, xlim=[0, 1], ylim=[0, 1], marker='.', color='darkblue',
               plot_central_tendency=True, central_tendency=np.median):

    def _format_ticks(lim, extra_values=None):
        vals = []
        if lim[0] < 0:
            vals = [lim[0], 0]
        else:
            vals = [lim[0]]
        if lim[1] > 1:
            vals.extend([lim[1], 1])
        else:
            vals.append(lim[1])
        if extra_values:
            try:
                vals.extend(extra_values)
            except:
                vals.append(extra_values)
        vals = np.sort(vals)

        labels = ['%0.2f' % v for v in vals]
        labels = ['0' if l == '0.00' else l for l in labels]
        labels = ['1' if l == '1.00' else l for l in labels]

        return vals, labels

    valid_idxes = np.where(np.isfinite(x) & np.isfinite(y))[0]
    x = np.array(x)[valid_idxes]
    y = np.array(y)[valid_idxes]
    plt.scatter(x, y, marker=marker, color=color)
    plt.plot(xlim, ylim, 'k', linewidth=.75)
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if plot_central_tendency:
        ct_x = central_tendency(x)
        plt.plot([ct_x, ct_x], ylim, color='darkorange')
        ct_y = central_tendency(y)
        plt.plot(xlim, [ct_y, ct_y], color='darkorange')
        plt.scatter(ct_x, ct_y, color='darkorange')

        vals, labels = _format_ticks(xlim, ct_x)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim, ct_y)
        plt.yticks(vals, labels)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

class FigViewer():

    def __init__(self, fig_dir='.'):
        self.fig_dir = Path(fig_dir)
        self.file_list = None
        self.get_file_list()
        self.file_idx = 0
        self.output = widgets.Output()
        self.button = widgets.Button(description='Next')
        self.button.on_click(self.on_button_clicked)
        display(self.output)
        display(self.button)
        self.show()

    def get_file_list(self):
        suffixes = ('.svg', 'jpg')
        files = list(Path(self.fig_dir).iterdir())
        files.sort()
        files = [f for f in files if (f.suffix == '.svg')]
        self.file_list = [f for f in files if f.suffix in suffixes]

    def show(self):
        with self.output:
            display(HTML(open(self.file_list[self.file_idx]).read()))
            clear_output(wait=True)
            self.file_idx = self.file_idx + 1

    def on_button_clicked(self, _):
        self.show()
