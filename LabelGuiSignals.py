from LabelGui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from utils import *


# 在这里定义所有的signals, slots
class DisplayFeature(QMainWindow, Ui_MainWindow):

    def __init__(self, defectclass, params):
        super(DisplayFeature, self).__init__()
        self.setupUi(self)
        # self.CurMeangraphicsView.setBackground(background='w')
        # self.features = data  # ndarray of the features, which need to be displayed in the Graphview
        self.features = np.array([])
        self.score_analys = np.array([])
        self.bad_features = np.array([])
        self.params = params
        self.threshold = self.params['threshold']
        # self.displaytext = displaytext
        self.displaytext = ''
        self.rect = 0
        self.class_selected = 0
        self.rect_region = []
        self.color_dict = ['FFB90F', '2E8B57', 'FF00FF', '836FFF', '008B8B', '0000FF']  # color of each class ROI rect
        self.defect_class = defectclass
        self.save_flag = False
        self.idle_stop = 0
        self.idle_start = 0

        self.initUI()

    def initUI(self):
        # put all signals and slots here
        self.setWindowTitle('Feature Display')
        self.setWindowIcon(QtGui.QIcon('labelGuiIcon.png'))
        # self.OpenFileButton.setDisabled(True)  # This will hide the open file button, make sure this command is needed
        self.Class1radioButton.setCheckable(False)
        self.Class2radioButton.setCheckable(False)
        self.Class3radioButton.setCheckable(False)
        self.Class4radioButton.setCheckable(False)
        self.Class5radioButton.setCheckable(False)
        self.Class6radioButton.setCheckable(False)

        self.LabelDatacheckBox.setChecked(False)
        self.AddRectpushButton.setDisabled(True)
        # self.SaveROIRectpushButton.setDisabled(True)

        self.set_RadioButtontext()
        # self.set_text_box()
        # self.plot_data()
        # self.set_table()

        self.OpenFileButton.clicked.connect(self.open_dir)
        self.FinishiButton.clicked.connect(self.save_rect)
        self.FinishiButton.clicked.connect(self.save_table)
        self.FinishiButton.clicked.connect(self.transfer_params)
        self.FinishiButton.clicked.connect(self.save_feature_to_file)
        self.FinishiButton.clicked.connect(self.close)

        self.LabelDatacheckBox.clicked.connect(self.select_class)
        self.AddRectpushButton.clicked.connect(self.add_rect)
        # self.SaveROIRectpushButton.clicked.connect(self.save_rect)
        self.Class1radioButton.clicked.connect(self.set_rect_color)
        self.Class2radioButton.clicked.connect(self.set_rect_color)
        self.Class3radioButton.clicked.connect(self.set_rect_color)
        self.Class4radioButton.clicked.connect(self.set_rect_color)
        self.Class5radioButton.clicked.connect(self.set_rect_color)
        self.Class6radioButton.clicked.connect(self.set_rect_color)
        # self.BadDatatableWidget.itemChanged.connect(self.save_table)


    def open_dir(self):
        filename = QtWidgets.QFileDialog.getOpenFileName()
        if len(filename[0]) != 0:
            self.displaytext = filename[0]
            self.set_text_box()
            print('{} is selected'.format(self.displaytext))
            self.features, self.score_analys, self.idle_stop, self.idle_start = feature_extraction(self.displaytext)
            self.plot_data()
            self.set_table()
        else:
            self.close()
            print('None file selected')

    def set_text_box(self):
        self.DisplaytextBrowser.clear()
        self.DisplaytextBrowser.setText(self.displaytext)

    def plot_data(self):
        self.ScoresAnalysgraphicsView.addLegend((1, -40))
        self.ScoresAnalysgraphicsView.plot(self.score_analys[:, 0], self.score_analys[:, 1], pen=pg.mkPen(0, 100, 100),
                                     symbolSize=4, symbolBrush=pg.mkBrush(50, 0, 250, 120), antialis=True, name='score mean')
        self.ScoresAnalysgraphicsView.plot(self.score_analys[:, 0], self.score_analys[:, -1], pen=pg.mkPen(255, 0, 100),
                                     symbolSize=4, symbolBrush=pg.mkBrush(50, 0, 250, 120), antialis=True, name='score std')

        self.ScoresgraphicsView.plot(self.features[:, 0], self.features[:, 24], pen=pg.mkPen(255, 100, 100),
                                     symbolSize=4, symbolBrush=pg.mkBrush(255, 100, 100), antialis=True)  # (50, 0, 250, 120)
        self.ScoresgraphicsView.addItem(pg.InfiniteLine(pos=self.params['threshold'], angle=0))

        self.CurMeangraphicsView.plot(self.features[:, 0], self.features[:, 2], pen=(0, 191, 255), antialis=True)
        self.CurStdgraphicsView.plot(self.features[:, 0], self.features[:, 3], pen=(0, 191, 255), antialis=True)
        self.CurMaxgraphicsView.plot(self.features[:, 0], self.features[:, 4], pen=(0, 191, 255), antialis=True)
        self.CurMingraphicsView.plot(self.features[:, 0], self.features[:, 5], pen=(0, 191, 255), antialis=True)
        self.CurPeakMeangraphicsView.plot(self.features[:, 0], self.features[:, 6], pen=(0, 191, 255), antialis=True)
        self.CurPeakStdgraphicsView.plot(self.features[:, 0], self.features[:, 7], pen=(0, 191, 255), antialis=True)
        self.CurPeakCountgraphicsView.plot(self.features[:, 0], self.features[:, 8], pen=(0, 191, 255), antialis=True)
        self.CurPeakWidthMeangraphicsView.plot(self.features[:, 0], self.features[:, 9], pen=(0, 191, 255), antialis=True)
        self.CurPeakWidthStdgraphicsView.plot(self.features[:, 0], self.features[:, 10], pen=(0, 191, 255), antialis=True)
        self.CurPeakGapMeangraphicsView.plot(self.features[:, 0], self.features[:, 11], pen=(0, 191, 255), antialis=True)
        self.CurPeakGapStdgraphicsView.plot(self.features[:, 0], self.features[:, 12], pen=(0, 191, 255), antialis=True)

        self.VolMeangraphicsView.plot(self.features[:, 0], self.features[:, 13], pen=(0, 191, 255), antialis=True)
        self.VolStdgraphicsView.plot(self.features[:, 0], self.features[:, 14], pen=(0, 191, 255), antialis=True)
        self.VolMaxgraphicsView.plot(self.features[:, 0], self.features[:, 15], pen=(0, 191, 255), antialis=True)
        self.VolMingraphicsView.plot(self.features[:, 0], self.features[:, 16], pen=(0, 191, 255), antialis=True)
        self.VolPeakMeangraphicsView.plot(self.features[:, 0], self.features[:, 17], pen=(0, 191, 255), antialis=True)
        self.VolPeakStdgraphicsView.plot(self.features[:, 0], self.features[:, 18], pen=(0, 191, 255), antialis=True)
        self.VolPeakCountgraphicsView.plot(self.features[:, 0], self.features[:, 19], pen=(0, 191, 255), antialis=True)
        self.VolPeakWidthMeangraphicsView.plot(self.features[:, 0], self.features[:, 20], pen=(0, 191, 255), antialis=True)
        self.VolPeakWidthStdgraphicsView.plot(self.features[:, 0], self.features[:, 21], pen=(0, 191, 255), antialis=True)
        self.VolPeakGapMeangraphicsView.plot(self.features[:, 0], self.features[:, 22], pen=(0, 191, 255), antialis=True)
        self.VolPeakGapStdgraphicsView.plot(self.features[:, 0], self.features[:, 23], pen=(0, 191, 255), antialis=True)


    def select_class(self):
        flag = self.LabelDatacheckBox.checkState()
        self.Class1radioButton.setCheckable(flag)
        self.Class2radioButton.setCheckable(flag)
        self.Class3radioButton.setCheckable(flag)
        self.Class4radioButton.setCheckable(flag)
        self.Class5radioButton.setCheckable(flag)
        self.Class6radioButton.setCheckable(flag)
        self.AddRectpushButton.setDisabled(not flag)
        # self.SaveROIRectpushButton.setDisabled(not flag)

    def draw_rect(self):
        if self.LabelDatacheckBox.isChecked():
            # self.items += 1
            # self.ScoresgraphicsView.mousePressEvent()
            self.rect = pg.RectROI([0, 0.5], [5, 0.25], pen={'color': "FFB90F", 'width': 1})
            self.rect.addScaleHandle([0, 0],[1, 1])
            self.rect.addScaleHandle([0, 1], [1, 0])
            self.rect.addScaleHandle([1, 0], [0, 1])
            self.ScoresgraphicsView.addItem(self.rect)
            self.LabelDisplaytextBrowser.setText(
                'rectangular left bottom={}, size={}'.format(self.rect.saveState()['pos'], self.rect.saveState()['size']))
            self.rect.sigRegionChangeFinished.connect(self.update_rect)
            # self.rect.sigRemoveRequested.connect(self.handle_removeROI)
        else:
            self.AddRectpushButton.setDisabled(True)
            # self.SaveROIRectpushButton.setDisabled(True)


    def update_rect(self):
        self.LabelDisplaytextBrowser.setText(
            'rectangular left bottom={}, size={}'.format(self.rect.saveState()['pos'], self.rect.saveState()['size']))
        # print('rectangular left bottom={}, size={}'.format(self.rect.saveState()['pos'], self.rect.saveState()['size']))

    def add_rect(self):
        self.save_rect()
        self.draw_rect()

    def save_rect(self):
        if type(self.rect)!=int:
            state = self.rect.saveState()
            state['class'] = self.class_selected
            self.rect_region.append(state)
            self.rect_region = [i for n, i in enumerate(self.rect_region) if i not in self.rect_region[n + 1:]]
            # print('{} ROI params saved'.format(len(self.rect_region)))

    def set_rect_color(self):
        if type(self.rect) != int:
            for i in range(6):
                var = 'self.Class{}radioButton'.format(i+1)
                if eval(var).isChecked():
                    self.class_selected = i
                    self.rect.setPen({'color': self.color_dict[i], 'width': 3})
                    break

    def transfer_params(self):
        self.rect_region = [i for n, i in enumerate(self.rect_region) if i not in self.rect_region[n + 1:]]
        self.LabelDisplaytextBrowser.setText(
            'Parameter transferred\n{}\nData from table modified!'.format(self.rect_region))
        print('ROI Parameter transferred\nData modified from table transferred')
        # self.close()
        # return self.rect_region

    def set_RadioButtontext(self):
        for i in range(len(self.defect_class)):
            eval('self.Class{}radioButton'.format(i+1)).setText(self.defect_class[i])

    def set_table(self):
        features_below_threshold = self.features[self.features[:, -2] < self.threshold]
        self.bad_features = np.stack(
            (features_below_threshold[:, 0], features_below_threshold[:, -2], features_below_threshold[:, -1]), axis=-1)

        self.BadDatatableWidget.setRowCount(self.bad_features.shape[0])
        self.BadDatatableWidget.setColumnCount(self.bad_features.shape[1])
        for r_idx, lst in enumerate(self.bad_features):
            for c_idx, value in enumerate(lst):
                self.BadDatatableWidget.setItem(r_idx, c_idx, QtGui.QTableWidgetItem(str(np.around(value, decimals=4))))

    def save_table(self):
        self.BadDatatableWidget.update()
        for i in range(self.BadDatatableWidget.rowCount()):
            if self.BadDatatableWidget.item(i, 2).text() != 0:
                self.bad_features[i, -1] = self.BadDatatableWidget.item(i, 2).text()
        # print('Table label modifies saved')
        # return self.bad_features

    # def close_gui(self):
    #     self.close()

    def save_feature_to_file(self):
        self.save_flag = True
        # save_feature(self.features,
        #              feature_csv=self.params['feature csv file'] +
        #                          '/feature_{}'.format(self.displaytext.rsplit('/', 1)[-1]))



