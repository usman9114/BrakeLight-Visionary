import wx
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from os.path import join
from os import listdir
import argparse
from src import brake_light
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class VideoPanel(wx.Panel):
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, -1, size=size)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.parent = parent
        self.SetDoubleBuffered(True)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.parent.bmp:
            dc.DrawBitmap(self.parent.bmp, 0, 0)

class MyFrame(wx.Frame):
    def __init__(self, drive_id, set_smooth=True, speed=2, video_data=None):
        wx.Frame.__init__(self, None)
        self.bmp = None
        self.set_smooth = bool(set_smooth)
        self.root_dir = video_data
        self.video_data = listdir(video_data)
        self.blight = BrakeLightDetect('night_light.csv')
        self.counter = 0

        acc = pd.read_csv('acceleration_zoox.csv', usecols=['x', 'y', 'z', 't', 'drive_id', 'together'])
        acc.index = pd.to_datetime(acc['t'], unit='s')

        acc_drive_data = acc[acc['drive_id'] == drive_id]
        accblank = acc_drive_data.copy()
        accblank['x'] = float('nan')
        accblank['y'] = float('nan')
        accblank['z'] = float('nan')
        accone = acc_drive_data.copy()
        accone['x'] = 0
        accone['y'] = 0
        accone['z'] = 0

        acc_drive_dataa = acc_drive_data[acc_drive_data['together'] == 1]
        acc_drive_dataa = acc_drive_dataa.append(accblank[acc_drive_data['together'] != 1])
        acc_drive_datan = acc_drive_data[acc_drive_data['together'] == 0]
        acc_drive_datan = acc_drive_datan.append(accblank[acc_drive_data['together'] != 0])
        acc_drive_datab = acc_drive_data[acc_drive_data['together'] == -1]
        acc_drive_datab = acc_drive_datab.append(accblank[acc_drive_data['together'] != -1])
        acc_drive_databb = acc_drive_data[acc_drive_data['together'] == -2]
        acc_drive_databb = acc_drive_databb.append(accblank[acc_drive_data['together'] != -2])

        self.acc_drive_data = acc_drive_data.resample('100L').mean()
        self.acc_drive_dataa = acc_drive_dataa.resample('100L').mean()
        self.acc_drive_datan = acc_drive_datan.resample('100L').mean()
        self.acc_drive_datab = acc_drive_datab.resample('100L').mean()
        self.acc_drive_databb = acc_drive_databb.resample('100L').mean()

        if self.set_smooth:
            windowwidth = 5
            self.acc_drive_data['x_smooth'] = gaussian_filter(self.acc_drive_data['x'], windowwidth)
            self.acc_drive_data['y_smooth'] = gaussian_filter(self.acc_drive_data['y'], windowwidth)
            self.acc_drive_data['z_smooth'] = gaussian_filter(self.acc_drive_data['z'], windowwidth)

        frame = cv2.imread(join(self.root_dir, self.video_data[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        videopPanel = VideoPanel(self, (600, 500))

        self.videotimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnUpdateVidoe, self.videotimer)
        self.videotimer.Start(1000 / int(speed))

        self.graph = Figure()
        plottPanel = FigureCanvas(self, -1, self.graph)

        self.ax1 = self.graph.add_subplot(111)
        self.gyro_count = 5

        self.ax1.set_ylim([-.5, .5])
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(videopPanel)
        sizer.Add(plottPanel)
        self.SetSizer(sizer)
        self.Fit()
        self.Show(True)

    def OnUpdateVidoe(self, event):
        frame = cv2.imread(join(self.root_dir, self.video_data[self.counter]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.blight.detect(frame, self.video_data[self.counter])
        frame = cv2.resize(frame, (600, 500), interpolation=cv2.INTER_AREA)
        img_buf = wx.ImageFromBuffer(frame.shape[1], frame.shape[0], frame)
        self.bmp = wx.Bitmap(img_buf)

        if self.set_smooth:
            _replace = ''
        else:
            _replace = '_smooth'

        ax = self.acc_drive_dataa['x'][self.counter:self.gyro_count].values
        ay = self.acc_drive_datan['x'][self.counter:self.gyro_count].values
        az = self.acc_drive_datab['x'][self.counter:self.gyro_count].values
        azb = self.acc_drive_databb['x'][self.counter:self.gyro_count].values
        x = list(self.acc_drive_dataa.index[self.counter:self.gyro_count].values)

        self.ax1.set_ylim([-.5, .5])
        self.ax1.plot(x, ax, 'r', label='acceleration event')
        self.ax1.plot(x, ay, 'g', label='no linear acceleration event')
        self.ax1.plot(x, az, 'b', label='Braking event')
        self.ax1.plot(x, azb, 'y', label='Hard Braking event')

        self.ax1.set_title('linear acceleration', pad=10)

        self.graph.tight_layout()
        self.graph.autofmt_xdate()
        self.ax1.legend(fontsize=9)
        self.graph.canvas.draw()
        self.ax1.cla()

        self.gyro_count += 5
        self.counter += 1
        self.Refresh()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', default='8', help='used to fast forward frames')
    parser.add_argument('--v', default='night_light', help='video frames folder path')
    parser.add_argument('--s', default=False, help='apply gaussian smoothing')
    parser.add_argument('--d', default='8451e3f2-fd51-44c5-8588-d33276c7c11b', help='drive id must be in csv')

    args = parser.parse_args()
    app = wx.App(0)
    myframe = MyFrame(speed=args.speed, drive_id=args.d, video_data=args.v, set_smooth=args.s)
    app.MainLoop
