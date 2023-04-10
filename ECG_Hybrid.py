
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import numpy as np

import pandas as pd                                     # Constructor
import matplotlib.pyplot as plt                         # Constructor           # Plot Distribution
from matplotlib.font_manager import FontProperties      # Plot Distribution
from sklearn.decomposition import PCA                   # Partition
import heartpy as hp                                    # Constructor
import tensorflow as tf                                 # CNN-LSTM
from sklearn.model_selection import train_test_split    # Partition
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout  # CNN-LSTM
from tensorflow.keras import layers                     # CNN-LSTM
from tensorflow.keras import activations                # CNN-LSTM
from keras.layers import LSTM                           # CNN-LSTM
from sklearn.metrics import classification_report       # CNN-LSTM
from sklearn.metrics import confusion_matrix            # CNN-LSTM
import seaborn as sn                                    # CNN-LSTM
import pandas as pd                                     # CNN-LSTM
from matplotlib.font_manager import FontProperties      # CNN-LSTM              # Plot Distribution


class ECGhybrid:
    def __init__(self):
        # Load Data Set
        self.df_train = pd.read_csv("C:/Users/USER/PycharmProjects/Neural_Network/ECG_HeartbeatClassification/dataset/mitbih_train.csv", header=None)
        self.df_test = pd.read_csv("C:/Users/USER/PycharmProjects/Neural_Network/ECG_HeartbeatClassification/dataset/mitbih_test.csv", header=None)
        self.color_box = ['#0F9D1D', '#A748EE', '#4778E5', '#1BBFC3', '#E6672E']
        self.df_combine = pd.concat([self.df_train, self.df_test])
        # self.df_combine = self.df_train
        plt.style.use('seaborn-darkgrid')

    def pre_sampling(self, bin, type):
        status = "Pre-Sampling Data"
        # Check Data Balancing / Count Unique Rows
        self.df_combine[187] = self.df_combine[187].astype(int)
        equilibrium = self.df_combine[187].value_counts()
        print(type)
        print(equilibrium, "\n")
        if bin == 1:
            self.class_distribution(equilibrium, status, type)

    def post_sampling(self, bin, type):
        status = "Post-Sampling Data"
        # Accumulate individual Class data
        from sklearn.utils import resample
        n_samples = 20000
        df_1 = self.df_combine[self.df_train[187] == 1]
        df_2 = self.df_combine[self.df_train[187] == 2]
        df_3 = self.df_combine[self.df_train[187] == 3]
        df_4 = self.df_combine[self.df_train[187] == 4]
        df_0 = (self.df_combine[self.df_train[187] == 0]).sample(n=n_samples, random_state=42)

        # Up-Sampling
        df_1_up = resample(df_1, replace=True, n_samples=n_samples, random_state=123)
        df_2_up = resample(df_2, replace=True, n_samples=n_samples, random_state=124)
        df_3_up = resample(df_3, replace=True, n_samples=n_samples, random_state=125)
        df_4_up = resample(df_4, replace=True, n_samples=n_samples, random_state=126)

        # Local Concatenation
        self.df_train_balanced = pd.concat([df_0, df_1_up, df_2_up, df_3_up, df_4_up])
        equilibrium = self.df_train_balanced[187].value_counts()
        print(status)
        print(equilibrium, "\n")
        if bin == 1:
            self.class_distribution(equilibrium, status, type)

    def partition(self, bin):
        # Set Binary Conditions
        if bin == 1:
            # Train partition Selection
            X = self.df_train_balanced.iloc[:, 0:187]
            # X = hp.enhance_peaks(X, iterations=2)
            Y = self.df_train_balanced[187]
            # PCA
            combined_predictors = 50
            X = X.to_numpy()
            # pca = PCA(n_components=combined_predictors)
            # pca.fit(X)
            # x_pca = pca.transform(X)

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.1)

    def cnn_lstm(self, bin, epochs, title):
        # Reshape train and test data to (n_samples, 187, 1), where each sample is of size (187, 1)
        X_train = np.array(self.X_train).reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_test = np.array(self.X_test).reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        model = tf.keras.models.Sequential()

        if bin == 1:
            # Convolution Layer 1
            model.add(Conv1D(filters=32, kernel_size=(5,), padding='same', name="Conv_Layer_1",
                             activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(X_train.shape[1], 1)))
            # Convolution Layer 2
            model.add(Conv1D(filters=64, kernel_size=(5,), padding='same', name="Conv_Layer_2",
                             activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
            # Convolution Layer 3
            model.add(Conv1D(filters=64, kernel_size=(5,), padding='same', name="Conv_Layer_3",
                             activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
            # Convolution Layer 4
            model.add(Conv1D(filters=128, kernel_size=(5,), padding='same', name="Conv_Layer_4",
                             activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
            # Convolution Layer 5
            model.add(Conv1D(filters=128, kernel_size=(5,), padding='same', name="Conv_Layer_5",
                             activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
            # Max Pool & Dropout 1
            model.add(MaxPool1D(pool_size=(5,), strides=2, padding='same',  name="Max_Pool_Layer_1"))
            model.add(Dropout(0.45, name="Dropout_Layer_1"))
            # LSTM Network 1
            model.add(LSTM(210, return_sequences=True, name="LSTM_1"))
            # Max Pool & Dropout 2
            model.add(MaxPool1D(pool_size=(5,), strides=2, padding='same',  name="Max_Pool_Layer_2"))
            # model.add(Dropout(0.45, name="Dropout_Layer_2"))
            # LSTM Network 2
            model.add(LSTM(190, return_sequences=True, name="LSTM_2"))
            # Max Pool & Dropout 3
            model.add(MaxPool1D(pool_size=(5,), strides=2, padding='same', name="Max_Pool_Layer_3"))
            # model.add(Dropout(0.45, name="Dropout_Layer_3"))
            # Flatten Layer
            model.add(Flatten(name="Flatten_Layer"))
            # Del this
            # model.add(Dense(5, activation=activations.relu, name="FC_Layer_"))
            # Fully Connected Layer 1
            model.add(Dense(5, activation=activations.relu, name="FC_Layer_1"))
            # Fully Connected Layer 2
            model.add(Dense(5, activation=activations.softmax, name="FC_Layer_2"))
            # Optimizers
            opt = optimizers.Adam(0.001)
            # Early Stopping Callback
            es_cb = EarlyStopping(monitor='val_loss', patience=4)
            # Model Construction
            model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
            model.summary()
            history = model.fit(X_train, self.Y_train, epochs=epochs, validation_data=(X_test, self.Y_test),
                                validation_split=0.20, callbacks=[es_cb])
            print(history.history)
            print(history.history.keys())
            # visualizing Accuracy
            font = FontProperties()
            font.set_family('serif bold')
            font.set_style('oblique')
            font.set_weight('semibold')
            fig1 = plt.figure(figsize=(12, 12))
            ax0 = fig1.add_subplot(111)
            plt.plot(history.history['acc'], label='train')
            plt.plot(history.history['val_acc'], label='test')
            ax0.set_ylabel('Accuracy', fontproperties=font, fontsize=12)
            ax0.set_xlabel('Epochs', fontproperties=font, fontsize=12)
            plt.legend(loc='lower right')
            font.set_size('large')
            font.set_family('monospace')
            font.set_style('normal')
            font.set_weight('bold')
            plt.title(title, fontproperties=font, fontsize=16)
            plt.show()
            # Evaluation Measures
            predict = model.predict(X_test)
            print(classification_report(self.Y_test, np.argmax(predict, axis=1)))
            # Confusion Matrix
            cm = confusion_matrix(self.Y_test, np.argmax(predict, axis=1))
            print(cm)
            df_cm = pd.DataFrame(cm, index=[i for i in "NSVFU"],
                                 columns=[i for i in "NSVFU"])
            # Heat Map
            fig_heat = plt.figure(figsize=(12, 10))
            ax2 = sn.heatmap(df_cm, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', figure=fig_heat, fmt='.1g',
                             square=True, linewidths=2, linecolor='white')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, horizontalalignment='right')
            font.set_family('monospace')
            font.set_style('normal')
            font.set_weight('bold')
            plt.title("Confusion Matrix Heat Map", fontproperties=font, fontsize=16)
            plt.show()

    def add_gaussian_noise(self, signal):
        noise = np.random.normal(0, 0.05, 186)
        return signal + noise

    def class_distribution(self, equilibrium, status, type):
        labels = ['Non-Ectopic', 'Unknown', 'Ventricular', 'Supraventricular', 'Fusion']
        if type == 0:
            # Plot Pie Chart for Balanced Data
            fig1 = plt.figure(figsize=(9, 9))
            explode = (0.05, 0.05, 0.05, 0.05, 0.05)
            plt.pie(equilibrium, colors=self.color_box, labels=labels, autopct='%1.1f%%', startangle=240, pctdistance=0.85,
                    explode=explode, shadow=True)
            centre_circle = plt.Circle((0, 0), 0.60, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            font = FontProperties()
            font.set_family('serif bold')
            font.set_weight('semibold')
            plt.title('Class Distribution: ' + status, fontproperties=font, fontsize=14)
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()

    def ecg_graph(self, bin):
        leg = iter(['Normal', 'Unknown', 'Ventricular',
                    'Supraventricular ectopic Beats', 'Fusion Beats'])
        if bin == 1:
            fig, axes = plt.subplots(5, 1, figsize=(16, 11), constrained_layout=True)
            colors = iter(self.color_box)
            for i, ax in enumerate(axes.flatten()):
                ax.plot(self.df_train_balanced.iloc[i, :186].T, color=next(colors))
                ax.legend(next(leg))
            plt.savefig('ECG Heartbeat Sample.PNG', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
            plt.show()

    def ecg_graph_noise(self, bin):
        leg = iter(['Normal', 'Unknown', 'Ventricular',
                    'Supraventricular ectopic Beats', 'Fusion Beats'])
        if bin == 1:
            fig, axes = plt.subplots(5, 1, figsize=(16, 11), constrained_layout=True)
            colors = iter(self.color_box)
            for i, ax in enumerate(axes.flatten()):
                self.df_train_balanced.iloc[i, :186] = self.add_gaussian_noise(self.df_train_balanced.iloc[i, :186])
                ax.plot(self.df_train_balanced.iloc[i, :186].T, color=next(colors))
                ax.legend(next(leg))
            plt.savefig('ECG Heartbeat Sample.PNG', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
            plt.show()

    def histogram_2d(self, bin, class_num, min_val=5, size=70, title=''):
        if bin == 1:
            font = FontProperties()
            font.set_family('serif bold')
            font.set_style('oblique')
            font.set_weight('semibold')
            img = self.df_train_balanced.loc[self.df_train_balanced[187] == class_num].values
            img = img[:, min_val: size]
            img_flatten = img.flatten()

            final1 = np.arange(min_val, size)
            for _ in range(img.shape[0] - 1):
                tempo1 = np.arange(min_val, size)
                final1 = np.concatenate((final1, tempo1))
            print(len(final1))
            print(len(img_flatten))
            plt.hist2d(final1, img_flatten, bins=(80, 80), cmap=plt.cm.jet)
            plt.title(title + " Beats", fontproperties=font, size=10)
            fig_title = '2D Histogram-'+title+'.PNG'
            plt.savefig(fig_title, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)
            plt.show()


ecg = ECGhybrid()
ecg.pre_sampling(0, 0)      # Graph? Chart?
ecg.post_sampling(0, 0)      # Graph? Chart?
ecg.ecg_graph(0)            # Graph?
ecg.ecg_graph_noise(0)            # Graph?
ecg.histogram_2d(0, 4, title="Fusion")
ecg.partition(1)            # De-Noise?
ecg.cnn_lstm(1, 23, 'Noise Set without Dropouts')

