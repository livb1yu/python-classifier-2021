#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np
import os
import sys
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv1D,Input, Flatten, ZeroPadding1D, GlobalAveragePooling1D
from tensorflow.keras.layers import MaxPooling1D, Activation, Multiply, Add
from tensorflow.keras.layers import Activation,BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from helper_code import *
from wfdb import processing
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from numba import cuda
import random
from numba import cuda

from sklearn.preprocessing import MultiLabelBinarizer

twelve_lead_model_filename = '12_lead_model'
six_lead_model_filename = '6_lead_model'
three_lead_model_filename = '3_lead_model'
two_lead_model_filename = '2_lead_model'
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    # device = cuda.get_current_device()
    # device.reset()
    print('Finding header and recording files...')

    # header_files, recording_files = find_challenge_files(data_directory)
    snomed_scored = pd.read_csv("SNOMED_mappings.csv", sep=";")
    snomed_d = {}
    for i, row in snomed_scored.iterrows():
        snomed_d[row['SNOMED CT Code']] = row['Dx']

    rel_data, labels = (extract_relevant_data(data_directory, snomed_d))
    num_recordings = len(rel_data)
    recording_files= rel_data
    header_files = [load_header(x+'.hea') for x in rel_data]
    age = [get_age(anno) for anno in header_files]
    gender = [get_sex(anno) for anno in header_files]
    leads = [get_leads(anno) for anno in header_files]
    adc_gains = [get_adc_gains(header_files[i], leads[i]) for i in range(len(header_files))]
    baselines = [get_baselines(header_files[i], leads[i]) for i in range(len(header_files))]
    age_clean, gender_bin = import_gender_and_age(age, gender)
    lengths = [len(load_recording(rel_data[i])[0])/get_frequency(header_files[i]) for i in range(num_recordings)]
    d = {}
    for i in range(len(rel_data)):
            d[rel_data[i]] = [labels[i], get_frequency(header_files[i]), lengths[i], age[i], gender[i], adc_gains[i], baselines[i]]
    # data_df = pd.DataFrame.from_dict(d, orient='index', columns=['label', 'freq', 'length', 'age', 'gender', 'adc', 'baseline'])
    ag = np.array(list(zip(age_clean, gender_bin)))

    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    eqc0 = [x[0] for x in equivalent_classes]
    eqc1 = [x[1] for x in equivalent_classes]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] in eqc1:
                labels[i][j] = eqc0[eqc1.index(labels[i][j])]
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(list(labels))

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')
    classes = list(snomed_d.keys())
    # print(classes)
    classes = set()
    # for header_file in header_files:
    #     header = load_header(header_file)
    #     classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    data = []
    # labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = header_files[i]
        recording = load_recording(recording_files[i])

        # Get age, sex and root mean square of the leads.
        age, sex, freq = get_age(header), get_sex(header), get_frequency(header)

        if freq != 500:
            resampled_record = []
            for j in range(len(recording)):
                tmp = processing.resample_sig(recording[j], freq, 500)[0]
                if len(tmp) < 5000:
                    resampled_record.append(fill_zeros(tmp, 5000))
                else:
                    resampled_record.append(tmp[:5000])
            data.append(resampled_record)
        else:
            record = []
            for i in range(len(recording)):
                if len(recording[i]) < 5000:
                    record.append(fill_zeros(recording[i], 5000))
                else:
                    record.append(recording[i][:5000])
            data.append(record)

        # current_labels = get_labels(header)
        # for label in current_labels:
        #     if int(label) in classes:
        #         j = classes.index(int(label))
        #         labels[i, j] = 1
    small_X, small_y, small_ag = [], [], []
    for i in np.random.choice(len(data), 30000):
        small_X.append(data[i])
        small_y.append(labels[i])
        small_ag.append(ag[i])
    data = np.array(small_X)
    labels = np.array(small_y)
    small_ag = np.reshape(np.array(small_ag), [-1, 2, 1])
    print(labels.shape)
    # Train models.

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = sorted(twelve_leads)
    filename = os.path.join(model_directory, get_model_filename(leads))

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    features = np.reshape(features, [-1, 5000, len(leads)])
    print(features.shape)
    print(labels.shape)
    print(len(leads))
    input_dim = [5000, len(leads)]

    classifier = ensemble_model(input_dim, (2,1), num_classes)
    optimizer = Adam(lr=1e-3)
    classifier.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    classifier.fit([features, small_ag], labels, batch_size = 32, epochs = 100, validation_split = 0.3)
    save_model(filename, classifier)
    tf.config.list_physical_devices('GPU')
    # device = cuda.get_current_device()
    # device.reset()

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = sorted(six_leads)
    filename = os.path.join(model_directory, get_model_filename(leads))

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    features = np.reshape(features, [-1, 5000, len(leads)])
    print(len(leads))
    input_dim = [5000, len(leads)]

    classifier = ensemble_model(input_dim, (2, 1), num_classes)
    optimizer = Adam(lr=1e-3)
    classifier.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    classifier.fit([features, small_ag], labels, batch_size=32, epochs=100, validation_split=0.3)
    save_model(filename, classifier)
    tf.config.list_physical_devices('GPU')
    device = cuda.get_current_device()
    device.reset()

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = sorted(three_leads)
    filename = os.path.join(model_directory, get_model_filename(leads))

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    features = np.reshape(features, [-1, 5000, len(leads)])
    print(len(leads))
    input_dim = [5000, len(leads)]

    classifier = ensemble_model(input_dim, (2, 1), num_classes)
    optimizer = Adam(lr=1e-3)
    classifier.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    classifier.fit([features, small_ag], labels, batch_size=32, epochs=100, validation_split=0.3)
    save_model(filename, classifier)
    tf.config.list_physical_devices('GPU')
    # device = cuda.get_current_device()
    # device.reset()

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = sorted(two_leads)
    filename = os.path.join(model_directory, get_model_filename(leads))
 
    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    features = np.reshape(features, [-1, 5000, len(leads)])
    print(len(leads))
    input_dim = [5000, len(leads)]

    classifier = ensemble_model(input_dim, (2, 1), num_classes)
    optimizer = Adam(lr=1e-3)
    classifier.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    classifier.fit([features, small_ag], labels, batch_size=32, epochs=100, validation_split=0.3)
    save_model(filename, classifier)
    tf.config.list_physical_devices('GPU')
    # device = cuda.get_current_device()
    # device.reset()
################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename,classifier):
# Construct a data structure for the model and save it.
    classifier.save(filename)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return tf.keras.models.load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return tf.keras.models.load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return tf.keras.models.load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return tf.keras.models.load_model(filename)

# Generic function for loading a model.
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    return tf.keras.models.load_model(filename)

def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads)

################################################################################
#
# Running trained model functions
#
################################################################################

# # Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
# def run_twelve_lead_model(model, header, recording):
#     leads = twelve_leads
#     return run_model(model, leads, header, recording)
#
# # Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
# def run_six_lead_model(model, header, recording):
#     leads = six_leads
#     return run_model(model, leads, header, recording)
#
# # Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
# def run_three_lead_model(model, header, recording):
#     leads = three_leads
#     return run_model(model, leads, header, recording)
#
# # Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
# def run_two_lead_model(model, header, recording):
#     leads = two_leads
#     return run_model(model, leads, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    # tf.config.experimental.list_physical_devices('GPU')
    # device = cuda.get_current_device()
    # device.reset()
    # classes = model['classes']
    # leads = model['leads']
    classifier = model
    leads = sorted(get_leads(header))
    # Load features.
    num_leads = len(leads)
    freq = get_frequency(header)

    age = [get_age(header)]
    gender = [get_sex(header)]

    age_test, gender_test = import_gender_and_age(age, gender)
    ag_test = np.array(list(zip(age_test, gender_test)))
    ag_test = np.reshape(np.array(ag_test), [-1, 2, 1])



    record = []
    snomed_scored = pd.read_csv("SNOMED_mappings.csv", sep=";")
    snomed_d = {}
    for i, row in snomed_scored.iterrows():
        snomed_d[row['SNOMED CT Code']] = row['Dx']

    classes = list(snomed_d.keys())
    # data = np.zeros((len(leads), 4096), dtype=np.float32)

    if freq != 500:
        for i in range(len(recording)):
            tmp = processing.resample_sig(recording[i], freq, 500)[0]
            if len(tmp) < 5000:
                record.append(fill_zeros(tmp, 5000))
            else:
                record.append(tmp[:5000])
    else:
        for i in range(len(recording)):
            if len(recording) < 5000:
                record.append(fill_zeros(recording[i], 5000))
            else:
                record.append(recording[i][:5000])
    data = np.array(record)

    data = np.reshape(data, [-1, 5000, num_leads])

    # Predict labels and probabilities.
    # labels = classifier.predict(data)
    labels = np.round(model.predict([data, ag_test]), decimals=3).astype(int)
    probabilities = model.predict([data, ag_test])
    probabilities = np.asarray(probabilities, dtype=np.float32)

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms


def extract_relevant_data(dbloc, snomed_d):
    relevant_data = []
    db_dat = [x.split('.')[0] for x in os.listdir(dbloc) if 'mat' in x]
    labels = []
    for i in range(len(db_dat)):
        anno = load_header(dbloc+'/'+db_dat[i]+'.hea')
        label = get_labels(anno)
        test_label = []
        switch = False
        try:
            for j in range(len(label)):
                if int(label[j]) in snomed_d:
                    test_label.append(label[j])
                    switch = True
            if switch:
                relevant_data.append(dbloc+'/'+db_dat[i])
                labels.append(test_label)
        except:
            print(label)
    return relevant_data, labels

def data_extractor_snomed_filtered(dbloc):
    records = []
    annos = []
    freq_d = {}
    db_dat = [x.split('.')[0] for x in os.listdir(dbloc) if 'mat' in x]
    for i in range(len(db_dat)):
        annos.append(load_header(dbloc + '/' + db_dat[i] + '.hea'))
        labels = get_labels(annos)

        records.append(load_recording(dbloc + '/' + db_dat[i]))

    labels = [get_labels(y) for y in annos]
    ages = [get_age(y) for y in annos]
    sex = [get_sex(y) for y in annos]
    return records, labels, ages, sex

def fill_zeros(signal, length):
    if length < len(signal):
        signal = signal[:length]
    else:
        segment = random.randint(0,length-len(signal))
        r = [0]*(length-len(signal))
        front = r[0:segment]
        back = r[segment:]
        signal = list(front) + list(signal) + list(back)
    return signal


def augmentation_layer(X_input, layers):
    for layer in layers:
        X_input = layer(X_input)
    return X_input


def res_block(X, f, filters, stride, stage, block, flag):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    kf1, kf2 = f[0], f[1]

    X_shortcut = X
    print(X_shortcut.shape)
    X = Conv1D(filters=filters, kernel_size=kf1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.pad(X, [[0, 0, ], [7, 7], [0, 0]], "CONSTANT")

    print(X.shape)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    #     print(X.shape)

    X = Dropout(0.2)(X)
    #     print(X.shape)

    X = Conv1D(filters=filters, kernel_size=kf2, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #     print(X.shape)

    X = BatchNormalization(name=bn_name_base + '2b')(X)
    #     print(X.shape)
    X = se_block(X, filters)
    print(X.shape)
    if flag:
        channel = filters // 4
        #         pad_input_X = GlobalAveragePooling1D()(X_shortcut)
        pad_input_X = tf.pad(X_shortcut, [[0, 0, ], [0, 0], [channel, channel]], "CONSTANT")
    else:
        pad_input_X = X_shortcut
    #     print(pad_input_X.shape)

    X = Add()([X, pad_input_X])
    X = Activation('relu')(X)

    return X


def se_resnet(input_shape=(5000, 12), classes=24, augment=True):
    X_input = Input(input_shape)
    #     X = ZeroPadding1D(14)(X_input)
    #     if augment:
    #         X = gaussian_noise_layer(X_input, 0.4)
    X = Conv1D(64, 15, 2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = tf.pad(X, [[0, 0, ], [7, 7], [0, 0]], "CONSTANT")
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, 2, padding='same')(X)

    X = res_block(X, [15, 7], 64, 2, stage=1, block='b', flag=False)

    X = res_block(X, [15, 7], 64, 2, stage=2, block='c', flag=False)

    X = res_block(X, [15, 7], 128, 2, stage=3, block='d', flag=True)

    X = res_block(X, [15, 7], 128, 2, stage=4, block='e', flag=False)

    X = res_block(X, [15, 7], 256, 2, stage=5, block='f', flag=True)

    X = res_block(X, [15, 7], 256, 2, stage=6, block='g', flag=False)

    X = res_block(X, [15, 7], 512, 2, stage=7, block='h', flag=True)

    X = res_block(X, [15, 7], 512, 2, stage=8, block='i', flag=False)

    X = Flatten()(X)
    #     mod1 = Model(inputs=X_input, outputs = X)

    #     mod2 = Dense(10, activation='relu')(age_gender)
    #     mod2 = Dense(2, activation='relu')(mod2)
    #     mod2 = Model(inputs=age_gender, outputs=mod2)
    #     combined = mod1.output + [mod2.output]
    #     X = Dense(classes, activation='sigmoid', name='fc' +str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=X, name='se_resnet')

    return model

def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling1D()(in_block)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])


def feature_block(features):
    xf = Flatten()(features)
    xf = Dense(5, activation='relu')(features)
    return xf


def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def add_noise(X_input, noisy_X):
    return tf.concat([X_input, noisy_X], 0)


def fc_age_model(ag_input_shape):
    ag = Input(ag_input_shape)
    mod1 = Dense(10, activation='relu')(ag)
    mod1 = Dense(2, activation='relu')(mod1)
    mod1 = Flatten()(mod1)
    model = Model(inputs=ag, outputs=mod1)
    return model


def ensemble_model(input1_dim = (5000, 12), input2_dim = (2,1), classes=24):
    #     model_1 = se_resnet((5000, 12), 24, True)
    model_1 = se_resnet(input1_dim, 24)
    model_2 = fc_age_model(input2_dim)

    combined = tf.keras.layers.concatenate([model_1.output, model_2.output])
    print(combined.shape)
    z = Dense(24, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(
        combined)
    print(z.shape)
    model = Model(inputs=[model_1.input, model_2.input], outputs=z)
    return model


def CNN(input_dim, out_dim):
    tf.config.experimental_run_functions_eagerly(True)
    model = Sequential()

    model.add(Conv1D(128, kernel_size=8, activation='relu', input_shape=input_dim))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(128, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(256, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(256, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2, 2, padding='same'))

    model.add(Dropout(0.2, input_shape=[64, 1]))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(10, activation='linear'))
    model.add(Flatten())
    #     model.add(Dense(out_dim, activation='sigmoid'))

    return model

def clean_up_gender_data(gender):
    gender = np.asarray(gender)
    gender[np.where(gender == "Male")] = 0
    gender[np.where(gender == "Female")] = 1
    gender[np.where(gender == "NaN")] = 2
    gender[np.where(gender == 'Unknown')] = 2
    gender = gender.astype(np.int)
    return gender

def clean_up_age_data(age):
    age = np.asarray(age)
    age[np.where(age == "NaN")] = -1
    return age

def import_gender_and_age(age, gender):
    gender_binary = clean_up_gender_data(gender)
    age_clean = clean_up_age_data(age)
    print("gender data shape: {}".format(gender_binary.shape[0]))
    print("age data shape: {}".format(age_clean.shape[0]))
    return age_clean, gender_binary
