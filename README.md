# **Automated Detection of Cardiac Arrhythmia based on a Hybrid CNN-LSTM Network**
An automated prediction algorithm is constructed using various pre-processing strategies based on a Hybrid Convolutional Neural Network and Long-Short Term Memory (CNN-LSTM Network). 
</br>
[Conference Paper Link](https://link.springer.com/chapter/10.1007/978-981-16-8774-7_32)

## Abstract
Cardiac Arrhythmia is an irregular sequence of electrical impulses
which result in numerous shifts in heart rhythms. Such cardiac abnormalities
can be observed using a standard medical examination known as Electrocardiogram (ECG). However, with the drastic increase in heart-disease patients, interpreting such pulsations on ECG can be time-consuming and a challenging task.
Thus, the primary objective of this paper is to propose an automated system
based on a hybrid model which consists of an amalgamation of Convolutional
Neural Networks (CNN) and Long-Short Term Memory (LSTM) in order to accurately detect and classify several cardiac arrhythmia ailments. The model incorporates a feature selection algorithm, Principal Component Analysis (PCA),
that ingresses the new features into 14-layers deep one-dimensional CNNLSTM network. The experiment is conducted using PhysionNet’s MIT-BIH
and PTB Diagnostics datasets and multiple strategies have been contemplated
for evaluation purposes: firstly, using smooth ECG signals with filtered noise
and alternatively, using signals that encompass artificially generated noise
based on a Gaussian distribution. The proposed system achieved an accuracy of
99% with the denoised sets and 98% using the data with artificially generated
noise, exhibiting a consistent and robust generalization performance and possesses the potential to be used as an auxiliary tool to assist clinicians in arrhythmia diagnoses.

## Introduction & Objective
Arrhythmias can be categorized into non-ectopic (N), ventricular tachycardia (V),
supraventricular tachycardia (S), fusion (F) and unclassifiable (U) beats [7]. Table 1
demonstrates the five classifications of different ECG beats according to ANSI/AAMI
EC57. For appropriate diagnosis of the aforementioned arrhythmia disorders, supervision by Electrocardiogram (ECG) is employed to monitor the deviations in cardiac
muscle depolarization and repolarization for every cardiac cycle. Treatment of each
cardiac abnormality is non-identical and is incompatible with other classes. Therefore,
it is imperative for cardiologists to accurately interpret the ECG signals before the
administration of any medicament procedure. However, due to the exponential increase in the global population, especially adults having severe heart complications,
undertaking such tasks can be tedious and time-consuming.

The primary incentive of this paper is to present an automated predictive model for
ECG inspection, which will make signal classification much easier to comprehend
with precision. Many healthcare systems have adopted Deep learning (DL) algorithms
[8] to classify arrhythmias accurately. The Convolutional neural network (CNN) is an
example of one of the widely used algorithms in finding a suitable solution for various complex tasks and diagnoses in the medical field and has achieved great success.
Since the process of analyzing signals can be considered as a time series problem, the
long short-term memory (LSTM) undergoes various utilization as well, as it possesses
the capacity to remember and pretermit knowledge depending on the significance of
the processed information. The overall structure of the proposed model can be dissected into four phases: preprocessing, feature extraction, classification and assessment

## Feature Engineering
For this study, a
peak enhancement technique has been considered which attempts to normalize the
amplitude followed by an increase in the largest amplitude or R-peak by using linear
transformations relative to the other portions of the signal. Thus, guaranteeing that the
R-peaks are uninterrupted and undisturbed.

![alt text](https://github.com/shahriar-rahman/Automated-Detection-of-Cardiac-Arrhythmia/blob/main/Diagrams/SignalEngineering.PNG)

On the contrary, superimposing disturbance to the raw signal data
is also a logical concern. While clean, noise-filtered data is essential for evaluating a
model’s performance on a faultless condition, it is equally imperative to take into
consideration that in reality, there is bound to be a few sporadicity in the dataset. In
most critical cases, the quality of data can deteriorate due to the presence of noise.
Thus, it is also worth assessing the model’s accuracy for the worst-case scenario,
which would occur in practical circumstances. Furthermore, adding noise in the samples can also reduce the overfitting issue as the variance will be lower due to the increased training error. Therefore, a random value based on a Gaussian noise is generated on a scale of 0.05 to create more disruption to the signal.

## CNN-LSTM Network
The proposed model consists of five consecutive convolution layers, each with a single stride and a kernel size of five. Padding is applied to maintain the identical dimension as the input for the following convolution layers, which can be achieved by applying shorter length segments with zero padding. The number of filters for the first
layer is 32, while the second and third layer include 64 filters and lastly, 128 for the
fourth and fifth layer. A feature map can be obtained through the operation of convolving the input with the specified kernels or filters.

![alt text](https://github.com/shahriar-rahman/Automated-Detection-of-Cardiac-Arrhythmia/blob/main/Diagrams/CNN-LSTM.PNG)

Following each LSTM layer, another subsampling layer is implemented. The temporal sequence output of the individual LSTM layer is carried by the pooling layer
which then, applies temporal Max-pooling and guides it to the subsequent layers. A
flatten layer is used to perform a flattening operation that collapses the pooled feature
map into a vector to be supplied into a series of fully connected layers. The data is
compiled by extracting from the previous layer and then, process it by performing
SoftMax operation in the final layer of the model, where probabilities for each of the
classes are assigned to classify arrhythmia, which can be represented as 'N', 'S', 'V', 'F'
or 'U'. 

## Performance Analysis & Conclusion
![alt text](https://github.com/shahriar-rahman/Automated-Detection-of-Cardiac-Arrhythmia/blob/main/Diagrams/ResultsTabular.PNG)
This paper proposes
an efficient computer-aided diagnosis system that encompasses a feature selection
algorithm PCA and a classification technique based on CNN + LSTM network structure to accurately identify five fragments of arrhythmia ailments. One-dimensional
signals acquired from ECG Physiobank (PTB) database are synthesized to create multiple instances for evaluating the model’s performance under various circumstances.

![alt text](https://github.com/shahriar-rahman/Automated-Detection-of-Cardiac-Arrhythmia/blob/main/Diagrams/ConfusionMatrix.PNG)

The accuracy of using smooth and denoised signals achieved 99% while the accuracy
of using artificially generated noise signals achieved 98%, confirming a satisfactory
and consistent generalization performance and could be a convenient tool for assisting
clinicians to diagnose cardiovascular disease and reduce their workload significantly.

