'''
Plotting functions
'''

# pylint: disable=C0103, R0912, R0914

from pathlib import Path
import numpy as np
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
