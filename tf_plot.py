# editable transfer function editor
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from matplotlib import cm

class TFPlot(FigureCanvas):

    def __init__(self, parent=None, vol_window=None):
        self._data = None
        self._load_data()
        self._vol_window = vol_window
        self.fig = Figure()
        self.counts = None
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)

        self.update_plot()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)


    def update_plot(self):
        self.axes.clear()
        self.axes.set_xlim(left=0, right=1)
        self.axes.set_ylim(bottom=0, top=1)
        if self.counts is None:
            self.counts, self.bins = np.histogram(self._vol_window.volume_data.flatten(),
                                                  bins=20,
                                                  density=True)
            self.bins = (self.bins[1:] + self.bins[:-1]) / 2
        # self.axes.hist(self._vol_window.volume_data.flatten(), density=True, stacked=True)
        self.axes.hist(self.bins, bins=len(self.counts), weights=self.counts)
        self.axes.plot(np.linspace(0, 1, len(self._data)), self._data, 'go--')
        self.draw()


    def on_click(self, e):
        if e.xdata is not None and e.ydata is not None:
            idx = int(np.round(len(self._data)*e.xdata))
            print('point {} adjusted to {}'.format(idx, e.ydata))
            self._data[idx] = e.ydata
            self.update_plot()
            self._vol_window.update_texture()

    def get_tff_as_rgba(self):
        intensities = self._data
        rgba = cm.seismic(np.linspace(0, 1, len(intensities))).astype(np.float32)
        rgba[:,3] = intensities[:]
        return rgba

    def _load_data(self):
        #mocking loading from file
        # self._data = np.linspace(0, 1, 20, dtype=np.float32)
        self._data = np.zeros(20, dtype=np.float32)

class TFPlotWindow(QtWidgets.QMainWindow):

    def __init__(self, vol_window):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(900,600)
        self.setWindowTitle('Transfer function editor')

        self.sp = TFPlot(parent=self, vol_window=vol_window)
        self.setCentralWidget(self.sp)

    @property
    def tff(self):
        return self.sp.get_tff_as_rgba()

if __name__ == '__main__':
    pass