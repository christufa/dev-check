import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import plotly.graph_objects as go
import plotly 
from IPython.display import HTML, display

def chris_make_blobs(n_samples, P_Y, P_Xs):
    y = pd.Series(P_Y.rvs(n_samples))
    x = y.apply(lambda x: P_Xs[x].rvs())
    data = pd.DataFrame(x.to_list())
    data.columns = [f'x_{i + 1}' for i in range(len(data.columns))]
    data['y'] = y
    return data

def chris_simple_make_blobs(n_samples=100, centers=2, mean_range=(-10, 10), spread=0.25):
    data = pd.DataFrame({'y': np.random.randint(0, centers, n_samples)})
    means = []
    for i in range(centers):
        mask = data['y'] == i
        mean = np.random.random((2, )) * mean_range
        data.loc[mask, ['x_1', 'x_2']] = np.random.normal(mean, spread, (mask.sum(), 2))
        means.append(mean)


    return data, np.array(means)

def latex_helper():
    """vscode latex doesn't seem to work, just return None if it does or something like that"""
    ## Tomas Mazak's workaround
    plotly.offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))