from utils import *
import os
import argparse



# comment: argument parse setup
parser = argparse.ArgumentParser(description='Run incremental learning for current and voltage model')
parser.add_argument('state', type=str, default='val', help='set state of the model: train or val or test')
parser.add_argument('--file', nargs='+', default='', help='directory of file need to be val or test')
parser.add_argument('--log', type=str, default='learning_log.log', help='log file name, format: xxx.log')
args = parser.parse_args()

# pass parameters here instead of command line
# args = parser.parse_args(['test', '--file', r'D:\OneDrive - University of Wollongong\STEVENPAN\Vol_Data\NAB1\nab_layer6_3.csv'])


state = args.state
log = args.log
file_dir = [' '.join(args.file)]

# comment: logger
logger_setup(log)
logging.debug('-' * 50)
logging.debug('state -- {}'.format(state))

# comment: find number of files need to be processed
params = load_params('config.yaml')
defectClass = load_params('defect class config.yaml')

# comment: After label, feed into incremental learning model
try:
    with open('model_sklearn.pkl', 'rb') as f:
        init_model = pickle.load(f)
except FileNotFoundError:

    if state == 'train':
        # TODO: make sure all parameters presets in SGDClassifier are right and enough.
        init_model = SGDClassifier(loss='hinge',
                              penalty='l2',
                              alpha=0.01,
                              random_state=42,
                              learning_rate='optimal',  # learning rate should be tuned carefully. Use 'optimal' for best result
                              early_stopping=False)
    else:
        # print('No pre-trained model, cannot validate data')
        logging.debug('No pre-trained model, cannot validate or test data')


# comment: 2 dimentional SGD classifier, for plotting
try:
    with open('model_sklearn_plot.pkl', 'rb') as f:
        init_model_plot = pickle.load(f)
except FileNotFoundError:

    if state == 'train':
        # TODO: make sure all parameters presets in SGDClassifier are right and enough.
        init_model_plot = SGDClassifier(loss='hinge',
                              penalty='l2',
                              alpha=0.001,
                              random_state=42,
                              learning_rate='optimal',  # learning rate should be tuned carefully. Use 'optimal' for best result
                              early_stopping=False)
    else:
        # print('No pre-trained model, cannot validate data')
        logging.debug('No pre-trained model to display, cannot display validation or test data')


if state == 'train':
    CsvObserve = CsvFileObserve('config.yaml', state)
    flag, recursive_time = CsvObserve.Flag()
    file_dir = CsvObserve.cur_file
    file_dir.sort(reverse=True, key=os.path.getmtime)  # sort file from new to old
    file_dir = file_dir[:recursive_time]

    X_val = np.zeros([1, 24])
    y_val = np.zeros([1,])

    scaler = StandardScaler()

    # split train and validation subsets
    # set random_state to avoid data leakage
    file_dir, file_dir_val = sklearn.model_selection.train_test_split(file_dir, test_size=0.33, random_state=42)
    recursive_time = int(len(file_dir))

    # prepare validation data
    for j in range(len(file_dir_val)):
        val_csv = file_dir_val[j]
        val_data = np.asarray(pd.read_csv(val_csv))
        val_data = val_data.astype('float32')

        X_val_tmp, y_val_tmp = val_data[:, 1:25], val_data[:, -1]

        X_val = np.r_[X_val, X_val_tmp]
        y_val = np.r_[y_val, y_val_tmp]
    else:
        X_val, y_val = X_val[1:, :], y_val[1:, ]
        X_val, y_val = sklearn.utils.shuffle(X_val, y_val)

    # Standardize validate data
    X_val = scaler.fit_transform(X_val)


    # # ----------------------------for plot decision surface---------------------------------------
    # X_display = PCA_main(X_val, n_components=2)
    # y_display = y_val
    # # --------------------------------------------------------------------------------------------


    # these command is necessary or not?
    # print('\nFile processing:')
    logging.debug('Files processing:')
    if recursive_time != 0:
        # for i in range(recursive_time):
        #     # print(file_dir[i], '\n')
        #     logging.debug('{}\n'.format(file_dir[i]))
        logging.debug('{} data files are training\n'.format(recursive_time))
    else:
        # print('None')
        logging.debug('None data file is training')

    # initialize a dict to record training histories for each class, metrics includes: precision, recall, F1
    # save order:
    # 0-2: defect class 0 with metrics precision, recall and F1
    # 3-5: defect class 1 with metrics precision, recall and F1, and so on ...
    history = {i:[] for i in np.arange(len(defectClass) * 3)}

    fig = plt.figure()
    # fig, axs = plt.subplots(3, 3)  # setup screen
    fig.set_size_inches(16, 12)

    # comment: set state ='train' if you want update model
    for i in range(params['epoch']):

        # initialize
        data_display = None  # necessary???

        if i == 0:
            logging.debug('epoch {}/{}'.format(i + 1, params['epoch']))
            run_model, run_model_plot, confusion, data_display = \
                run_incremental_frame(init_model, init_model_plot, recursive_time, file_dir, (X_val, y_val), params, defectClass, state=state)
            history = history_record(confusion, history, defectClass)
        else:
            logging.debug('epoch {}/{}'.format(i + 1, params['epoch']))
            run_model, run_model_plot, confusion, data_display = \
                run_incremental_frame(run_model, run_model_plot, recursive_time, file_dir, (X_val, y_val), params, defectClass, state=state)
            history = history_record(confusion, history, defectClass)

            # plot decision surface
            if (i % (params['epoch'] // 9) == 0) & (i !=0):  # plot decision surface at multiple of 3
                ax = fig.add_subplot(3, 3, i // (params['epoch']//9))
                X_display, y_display = data_display
                X_display, y_display = X_display[1:, :-1], y_display[1:, ] # withdraw first row (all 0 in first row)
                # X_display = scale(X_display)

                # ax.set_title('Decision surface: epoch {}'.format(i))
                # decision_surface_plot(run_model_plot, ax, X_display, y_display,
                #                       target_names=[d for d in defectClass.values()],
                #                       colors='brygmckw')

                ax.set_title('Training data: epoch {}'.format(i))
                data_scatter_plot(ax, X_display, y_display,
                                  target_names=[d for d in defectClass.values()],
                                  colors='brygmckw')

    else:
        fig.tight_layout()
        plt.show()

        # plot training history
        plot_train_history(history, defectClass)

        # todo: you need to use run_model in the last run epoch to predict the validation dataset
        #  to provide the right confusion matrix and the final precision, recall, and f1 score

        # save model to disk
        save_model(run_model, 'model_sklearn.pkl')
        save_model(run_model_plot, 'model_sklearn_plot.pkl')



# comment: set state='val' if you want use model to validate data and no updates applied in model
if state == 'val':
    run_incremental_frame(init_model, init_model_plot, 1, file_dir, None, params, defectClass, state=state)


# comment: set state='test' if you want use model to predict data and no updates applied in model
# we handle raw data for testing
if state == 'test':
    final_feature, _, _, _ = feature_extraction(file_dir[0])
    preds_label = run_incremental_frame(init_model, init_model_plot, 1, file_dir, None,
                                        params, defectClass, state=state, array=final_feature)
    # print(final_feature.shape, preds_label.shape)
    predict_time = evaluate_prediction(final_feature, preds_label, criterion=params['threshold'])
    # optim_time = optim_prediction(predict_time, defectClass, pool_shift_threshold=0.2)  # optimize defect prediction results
    # optim_time = optim_prediction_simple_version(predict_time, defectClass)  # optimize defect prediction results

    # for defect, t in optim_time.items():
    for defect, t in predict_time.items():
        # logging.debug('')
        logging.debug('{}'.format('\n'))
        logging.debug('{} time stamp:'.format(defect))
        for stamp in t:
            # logging.debug('\t', '%.4f' % stamp, 's')
            logging.debug('{} {}s'.format('\t', '%.4f' % stamp))






