from .plot import *
from .describe import *

def visualize_data(path):
    describe(path, header=False)
    input("~~Enter To Continue~~")
    pair_plot(path, _class=1, header=False, column_range_features=list(range(2, 32)))
