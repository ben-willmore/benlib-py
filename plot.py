'''
Plotting functions
'''

# pylint: disable=C0103, R0912, R0914

from string import ascii_lowercase, ascii_uppercase
from textwrap import wrap
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
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

from scipy.stats import kruskal, wilcoxon

def nan_kruskal(x_data, y_data):
    valid_idx = np.isfinite(x_data+y_data)
    valid_x = [x for x,y in zip(x_data, y_data) if np.isfinite(x) and np.isfinite(y)]
    valid_y = [y for x,y in zip(x_data, y_data) if np.isfinite(x) and np.isfinite(y)]
    r,p = kruskal(valid_x, valid_y)
    return p

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

def _scatter_central_tendency(x, y, central_tendency=np.median,
                             central_tendency_color='darkorange',
                             central_tendency_lines_full_width=True):

    if rcParams['font.size'] < 10:
        marker_params = {'markersize': 12, 'lw': .75}
        marker_params_scatter = {'s':12, 'lw':.75}
    else:
        marker_params = {}
        marker_params_scatter = {}

    ct_x = central_tendency(x)
    ct_y = central_tendency(y)

    xlim = plt.xlim()
    ylim = plt.ylim()

    if central_tendency_lines_full_width:
        lim = ylim
    else:
        lim = [ylim[0], ct_y]
    plt.plot([ct_x, ct_x], lim, color=central_tendency_color, label='_nolegend_',
        **marker_params)

    if central_tendency_lines_full_width:
        lim = xlim
    else:
        lim = [xlim[0], ct_x]

    plt.plot(lim, [ct_y, ct_y], color=central_tendency_color, label='_nolegend_',
        **marker_params)
    plt.scatter(ct_x, ct_y, color=central_tendency_color, label='_nolegend_',
        **marker_params_scatter)

    vals, labels = _format_ticks(xlim, ct_x)
    plt.xticks(vals, labels)
    vals, labels = _format_ticks(ylim, ct_y)
    plt.yticks(vals, labels)

def _scatter_p_value(x, y, scatter_p_value_func=wilcoxon,
                     font_size=None):
    # r, p = kruskal(x, y)
    res = scatter_p_value_func(x, y)
    p = res.pvalue
    if p>.05:
        p = 'n.s.'
    elif p<0.001:
        p = 'p<.001'
    else:
        p = 'p='+('%.3f' % p).lstrip('0')

    if font_size is None:
        font_size = rcParams['font.size'] + 2
    label_bottom_right(p, size=font_size)

def scatter_cc(x, y, xlim=[0, 1], ylim=[0, 1], marker='.', color='tab:blue',
               plot_central_tendency=True, central_tendency=np.median,
               box=False,
               central_tendency_color='k',
               central_tendency_lines_full_width=None, # use box instead
               accept_nans=True,
               p_value=False, p_value_font_size=None):

    if central_tendency_lines_full_width is not None:
        box = central_tendency_lines_full_width

    if rcParams['font.size'] < 10:
        marker_params = {'markersize': 12, 'lw': .75}
        marker_params_scatter = {'s':12, 'lw':.75}
    else:
        marker_params = {}
        marker_params_scatter = {}

    x = np.array(x)
    y = np.array(y)

    if accept_nans:
        valid_idxes = np.where(np.isfinite(x) & np.isfinite(y))[0]
        x = x[valid_idxes]
        y = y[valid_idxes]
    else:
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError('NaN values present (you may want accept_nans=True)')

    plt.scatter(x, y, marker=marker, color=color, **marker_params_scatter)
    plt.plot(xlim, ylim, 'k', linewidth=.75)
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if plot_central_tendency:
        _scatter_central_tendency(x, y, central_tendency=central_tendency,
                                  central_tendency_color=central_tendency_color,
                                  central_tendency_lines_full_width=central_tendency_lines_full_width)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

    if p_value:
        _scatter_p_value(x, y, scatter_p_value_func=wilcoxon, font_size=p_value_font_size)

    if not box:
        plt.gca().spines[['right', 'top']].set_visible(False)

def scatter_cc_2(x1, y1, x2, y2,
                 color1='tab:blue', color2='tab:orange',
                 **kwargs):
    scatter_cc_multi([x1, x2], [y1, y2], colors=[color1, color2], **kwargs)

def scatter_cc_multi(x, y, xlim=[0, 1], ylim=[0, 1], markers='.+x', colors='tab:blue',
               plot_central_tendency=True, central_tendency=np.median,
               box=False,
               central_tendency_color='k',
               central_tendency_lines_full_width=None, # use box instead
               accept_nans=True,
               p_value=False, p_value_font_size=None):

    if rcParams['font.size'] < 10:
        marker_params = {'markersize': 12, 'lw': .75}
        marker_params_scatter = {'s':12, 'lw':.75}
    else:
        marker_params = {}
        marker_params_scatter = {}

    x = [np.array(d) for d in x]
    y = [np.array(d) for d in y]

    all_x = np.hstack(x)
    all_y = np.hstack(y)

    if accept_nans:
        valid_idxes = np.where(np.isfinite(all_x) & np.isfinite(all_y))[0]
        all_x = all_x[valid_idxes]
        all_y = all_y[valid_idxes]
    else:
        if np.any(np.isnan(all_x)) or np.any(np.isnan(all_y)):
            raise ValueError('NaN values present (you may want accept_nans=True)')

    for idx, (this_x, this_y) in enumerate(zip(x, y)):
        if accept_nans:
            valid_idxes = np.where(np.isfinite(this_x) & np.isfinite(this_y))[0]
            if len(valid_idxes)==0:
                continue
            this_x = this_x[valid_idxes]
            this_y = this_y[valid_idxes]

        if idx < len(markers):
            marker = markers[idx]
        else:
            marker = markers[0]
        if type(colors) in (str,tuple):
            color = colors
        elif idx < len(colors):
            color = colors[idx]
        else:
            color = colors[0]

        plt.scatter(this_x, this_y, marker=marker, color=color, **marker_params_scatter)

    plt.plot(xlim, ylim, 'k', linewidth=.75)
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if plot_central_tendency:
        _scatter_central_tendency(all_x, all_y, central_tendency=central_tendency,
                                  central_tendency_color=central_tendency_color,
                                  central_tendency_lines_full_width=central_tendency_lines_full_width)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

    if p_value:
        _scatter_p_value(all_x, all_y, scatter_p_value_func=wilcoxon, font_size=p_value_font_size)

    if not box:
        plt.gca().spines[['right', 'top']].set_visible(False)


def scatter_cc_old(x, y, xlim=[0, 1], ylim=[0, 1], marker='.', color='darkblue',
               plot_central_tendency=True, central_tendency=np.median,
               central_tendency_color='darkorange',
               central_tendency_lines_full_width=True,
               accept_nans=True,
               p_value=False, p_value_font_size=14):

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

    # new 20/09/2023
    x = np.array(x)
    y = np.array(y)

    if accept_nans:
        valid_idxes = np.where(np.isfinite(x) & np.isfinite(y))[0]
        x = x[valid_idxes]
        y = y[valid_idxes]
    else:
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError('NaN values present (you may want accept_nans=True)')

    # old
    # valid_idxes = np.where(np.isfinite(x) & np.isfinite(y))[0]
    # x = np.array(x)[valid_idxes]
    # y = np.array(y)[valid_idxes]

    plt.scatter(x, y, marker=marker, color=color)
    plt.plot(xlim, ylim, 'k', linewidth=.75)
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if plot_central_tendency:
        ct_x = central_tendency(x)
        ct_y = central_tendency(y)

        if central_tendency_lines_full_width:
            lim = ylim
        else:
            lim = [ylim[0], ct_y]
        plt.plot([ct_x, ct_x], lim, color=central_tendency_color, label='_nolegend_')

        if central_tendency_lines_full_width:
            lim = xlim
        else:
            lim = [xlim[0], ct_x]

        plt.plot(lim, [ct_y, ct_y], color=central_tendency_color, label='_nolegend_')
        plt.scatter(ct_x, ct_y, color=central_tendency_color, label='_nolegend_')

        vals, labels = _format_ticks(xlim, ct_x)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim, ct_y)
        plt.yticks(vals, labels)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

    if p_value:
        # r, p = kruskal(x, y)
        res = wilcoxon(x, y)
        p = res.pvalue
        if p>.05:
            p = 'n.s.'
        elif p<0.001:
            p = 'p<.001'
        else:
            p = 'p='+('%.3f' % p).lstrip('0')
        label_bottom_right(p, size=p_value_font_size)

    if not central_tendency_lines_full_width:
        plt.gca().spines[['right', 'top']].set_visible(False)


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

def add_panel_labels(axes=None, lowercase=False,
                 fontsize=None, fontweight='bold',
                 x_offset=-0.1, y_offset=1,
                 letter_offset=0):
    if axes is None:
        axes = np.array(plt.gcf().axes)
    if lowercase:
        letters = ascii_lowercase[letter_offset:]
    else:
        letters = ascii_uppercase[letter_offset:]

    sizes = [2,3,4,5,6,7,9,12,13,14,16,18]
    if rcParams['font.size']<=8:
        fontsize = 12
    else:
        fontsize = 16

    for idx, axis in enumerate(axes.reshape(-1)):
        axis.text(x_offset, y_offset, letters[idx],
                  transform=axis.transAxes,
                  fontsize=fontsize, fontweight=fontweight,
                  va='bottom', ha='right')

def add_caption(txt, wrap_width=100, fontsize=12):
    txt = '\n'.join(wrap(txt, width=wrap_width))
    plt.figtext(0.12, 0.01, txt,  fontsize=fontsize, va='top', ha='left')

def scatter_cc_2_old(x1, y1, x2, y2, xlim=[0, 1], ylim=[0, 1], link_points=False,
                 color1='red', color2='blue',
                 plot_central_tendency=True, central_tendency=np.median,
                 central_tendency_color='darkorange',
                 central_tendency_lines_full_width=True):

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

    valid_idxes = np.where(np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y1) & np.isfinite(y2))[0]
    x1 = np.array(x1)[valid_idxes]
    x2 = np.array(x2)[valid_idxes]
    y1 = np.array(y1)[valid_idxes]
    y2 = np.array(y2)[valid_idxes]
    if link_points:
        for x1p, x2p, y1p, y2p in zip(x1,x2,y1,y2):
            plt.plot([x1p, x2p], [y1p, y2p], color='lightgrey', linewidth=0.75, label='_nolegend_')
    plt.scatter(x1, y1, marker='+', color=color1, zorder=2)
    plt.scatter(x2, y2, marker='.', color=color2, zorder=2)
    plt.plot(xlim, ylim, 'k', linewidth=.75, label='_nolegend_')
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)

    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))

    if plot_central_tendency:
        ct_x = central_tendency(x)
        ct_y = central_tendency(y)

        if central_tendency_lines_full_width:
            lim = ylim
        else:
            lim = [ylim[0], ct_y]
        plt.plot([ct_x, ct_x], lim, color=central_tendency_color, label='_nolegend_')

        if central_tendency_lines_full_width:
            lim = xlim
        else:
            lim = [xlim[0], ct_x]

        plt.plot(lim, [ct_y, ct_y], color=central_tendency_color, label='_nolegend_')
        plt.scatter(ct_x, ct_y, color=central_tendency_color, label='_nolegend_', zorder=2)

        vals, labels = _format_ticks(xlim, ct_x)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim, ct_y)
        plt.yticks(vals, labels)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

    if not central_tendency_lines_full_width:
        plt.gca().spines[['right', 'top']].set_visible(False)

def choose_panel_layout(n):
    factor_pairs = []
    for i in range(1, int(np.sqrt(n)+1)):
        if n % i == 0:
            factor_pairs.append((n//i, i))
    factor_pairs.sort(key=lambda p:p[0]-p[1])
    return factor_pairs[0]

def remove_frame(ax):
    ax.spines[['right', 'top']].set_visible(False)

def remove_frames(fig=None):
    if fig is None:
        ax = plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
    else:
        for ax in fig.axes:
            ax.spines[['right', 'top']].set_visible(False)

def label_top_left(text, margin=.03, outside=False, ax=None, size=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    xpos = xlim[0] + (xlim[1]-xlim[0])*margin
    ylim = ax.get_ylim()

    if outside:
        ypos = ylim[0] + (ylim[1]-ylim[0])*(1+margin)
        va = 'bottom'
    else:
        ypos = ylim[0] + (ylim[1]-ylim[0])*(1-margin)
        va = 'top'
    if size is None:
        size = rcParams['font.size']+2
    ax.text(xpos, ypos, text, ha='left', va=va, size=size)

def label_top_right(text, margin=.03, ax=None, size=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    xpos = xlim[0] + (xlim[1]-xlim[0])*(1-margin)
    ylim = ax.get_ylim()
    ypos = ylim[0] + (ylim[1]-ylim[0])*(1-margin)
    if size is None:
        size = rcParams['font.size']+2
    ax.text(xpos, ypos, text, ha='right', va='top', size=size)

def label_bottom_right(text, margin=.03, ax=None, size=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    xpos = xlim[0] + (xlim[1]-xlim[0])*(1-margin)
    ylim = ax.get_ylim()
    ypos = ylim[0] + (ylim[1]-ylim[0])*margin
    if size is None:
        size = rcParams['font.size']+2
    ax.text(xpos, ypos, text, ha='right', va='bottom', size=size)
