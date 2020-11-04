from utils import *
from PyQt5.QtWidgets import QApplication
from LabelGuiSignals import DisplayFeature


# comment: run data label GUI
def run_gui(defect_class, params):
    app = QApplication(sys.argv)
    mainWindow = DisplayFeature(defect_class, params)
    # sys.exit(app.exec_())
    mainWindow.show()
    app.exec_()
    return mainWindow.rect_region, mainWindow.bad_features, mainWindow.features, mainWindow.displaytext, mainWindow.save_flag, mainWindow.idle_stop, mainWindow.idle_start


# comment: conclude labels from GUI, and save feature to file
def generate_labelData(defect_class, params):

    # TODO: based on the scores we get in features, return bad data and require manually label (A GUI may needed?) #
    ROI_region, bad_features_labels, final_features, file_dir, save_flag, idle_stop, idle_start = run_gui(defect_class, params)
    print('Label index generated, GUI closed\n')

    for j in range(len(ROI_region)):
        idex_start = np.floor(
            (ROI_region[j]['pos'][0] - params['arc start time'] - idle_stop) / (params['data_time'] * params['window_len']))
        idex_stop = np.ceil(
            (ROI_region[j]['pos'][0] + ROI_region[j]['size'][0] - params['arc start time'] - idle_stop) / (
                        params['data_time'] * params['window_len']))  # plus 1 to get the data of stop index
        if idex_start < 0:
             idex_start = 0
        if idex_stop > final_features.shape[0]:
            idex_stop = final_features.shape[0]
        final_features[int(idex_start): int(idex_stop), -1] = int(ROI_region[j]['class'])
    else:
        print('{} ROI region labels modified'.format(len(ROI_region)))

    if len(bad_features_labels) != 0:
        if bad_features_labels[:, 2].any() == 0:
            print('no data labels modified from table')
        else:
            labels = bad_features_labels[bad_features_labels[:, 2] != 0]
            for j, t in enumerate(labels[:, 0]):
                labels_index = np.where(final_features[:, 0] == t)
                final_features[labels_index, -1] = labels[j, -1]
            print('{} data labels modified from table'.format(labels.shape[0]))
    else:
        print('no data generated')
    print()

    if save_flag:
        save_feature(final_features,
                     feature_csv=params['feature csv file'] +
                                 '/feature_{}'.format(file_dir.rsplit('/', 1)[-1]))

    # return format(file_dir.rsplit('/', 2)[0] + '/{}'.format(params['feature csv file']) +
    #               '/feature_{}'.format(file_dir.rsplit('/', 1)[-1]))


# comment: check a directory, if a new raw data generated, call feature extraction
param = load_params('config.yaml')
defectClass = load_params('defect class config.yaml')

# comment: generate labeled csv file with gui, save to -->
# comment: D:\Python Projects\Monitoring Project\Current_voltage_anomaly_detection\LabelData\feature_rawfilename.csv
generate_labelData(defectClass, param)

# if __name__ == '__main__':
#     feature_extraction('layer3_9.csv')
