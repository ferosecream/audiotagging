#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import math
import os
import librosa
import utils
import json
import pickle
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
TR_FEATURE_NPY = os.path.join(os.path.dirname(__file__),"../data/f1_{0}.npy")
TR_LABEL_NPY = os.path.join(os.path.dirname(__file__),"../data/f2_{0}.npy")
TS_FEATURE_NPY = os.path.join(os.path.dirname(__file__),"../data/f3_{0}.npy")
TS_F_NAME_NPY = os.path.join(os.path.dirname(__file__),"../data/f4_{0}.npy")
LABEL_DICT_NPY = os.path.join(os.path.dirname(__file__),"../data/f5.txt")

CHUNK_SIZE = 500
FEATURE_SIZE = 193

SAMPLE_RATE = 44100 # number of samples per second
N_MFCC = 40 # size of mfcc array returned by librosa
HOP_LENGTH = 512 # number of samples between successive frames
AUDIO_DURATION = 2 # 2 seconds
AUDIO_LENGTH = 1 + int(np.floor(AUDIO_DURATION*SAMPLE_RATE/HOP_LENGTH))
INPUT_LENGTH = SAMPLE_RATE * AUDIO_DURATION
iter = 1
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   extract_features()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Extracts features from an audio file using the librosa library.                             #
#                                                                                               #
#***********************************************************************************************#
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    try:
        stft = np.abs(librosa.stft(X))
    except:  # catch *all* exceptions
        print("Exception Catched : {0}".format(file_name))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=N_MFCC).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   p_train_mnn_thread()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse training audio files in multi-threaded environment.              #
#                                                                                               #
#***********************************************************************************************#
def p_train_mnn_thread(audio_path, label_dictionary, data):
    # initialize variables
    features, labels = np.empty((0,FEATURE_SIZE)), np.empty(0)
    # process this threads share of workload
    for i in range(data.shape[0]):
            # add a log message to be displayed after processing every 250 files.
            if i % int(CHUNK_SIZE/4) == 0:
                utils.write_log_msg("FEATURE_TRAIN - {0}...".format(i))
            line = data.iloc[i]
            fn = audio_path+line["fname"]
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label_dictionary[line["label"]])
    # return the extracted features to the calling program
    return features, labels

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   p_predict_mnn_thread()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse prediction audio files in multi-threaded environment.            #
#                                                                                               #
#***********************************************************************************************#
def p_predict_mnn_thread(audio_path, name_list):
    #global iter
    # initialize variables
    features = np.empty((0,FEATURE_SIZE))
    # traverse through the name list and process this threads workload
    for fname in name_list:
        # add a log message to be displayed after processing every 250 files.
        if len(features) % int(CHUNK_SIZE/4) == 0:
            utils.write_log_msg("FEATURE_PREDICT - {0}...".format(len(features)))
        #X, sample_rate = librosa.load(audio_path+fname, res_type='kaiser_fast')
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(audio_path+fname)
        #iter = iter + 1;
        #print("{0} : {1}".format(audio_path+fname, iter))
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
    # return the extracted features to the calling program
    return features

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   parse_audio_files_predict()                                                                 #
#                                                                                               #
#   Description:                                                                                #
#   Parses audio data that needs to be predicted upon.                                          #
#                                                                                               #
#***********************************************************************************************#
def parse_audio_files_predict(audio_path, name_list, nn_type, file_ext="*.wav"): 
    # create a thread pool to process the workload
    thread_pool = [] 

    NO_OF_CPUS = os.cpu_count()
    CHUNK_SIZE = int(math.ceil(len(name_list)/NO_OF_CPUS));

    # split the filename list into chunks of 'CHUNK_SIZE' files each
    data = utils.generate_chunks(name_list, CHUNK_SIZE) 
    # each chunk is the amount of data that will be processed by a single thread
    for chunk in data:
        if nn_type == 0:
            features = np.empty((0, FEATURE_SIZE))
            thread_pool.append(utils.ThreadWithReturnValue(target=p_predict_mnn_thread, args=(audio_path, chunk)))
        else:
            features = np.empty(shape=(0, N_MFCC, AUDIO_LENGTH, 1))
            thread_pool.append(utils.ThreadWithReturnValue(target=p_predict_cnn_thread, args=(audio_path, chunk)))
    # print a log message for status update
    utils.write_log_msg("PREDICT: creating a total of {0} threads...".format(len(thread_pool)))  
    # start the entire thread pool
    for single_thread in thread_pool:
        single_thread.start()
    # wait for thread pool to return their results of processing
    for single_thread in thread_pool:
        ft = single_thread.join()
        features = np.vstack([features,ft])
    # perform final touches to extracted arrays
    features = np.array(features)
    
    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # return the extracted features to the calling program
    return features, name_list

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   parse_audio_files_train()                                                                   #
#                                                                                               #
#   Description:                                                                                #
#   Parses the audio data that is to be used for training.                                      #
#                                                                                               #
#***********************************************************************************************#
def parse_audio_files_train(audio_path, train_csv_path, label_dictionary, nn_type, file_ext="*.wav"):
    # initialize variables
    labels = np.empty(0)  
    # read audio files using pandas and split it into chunks of 'CHUNK_SIZE' files each
    data = pd.read_csv(train_csv_path, chunksize=CHUNK_SIZE)
    # create a thread pool to process the workload
    thread_pool = []
    # each chunk is the amount of data that will be processed by a single thread
    for chunk in data:
        if(nn_type == 0):
            features = np.empty((0,FEATURE_SIZE))
            thread_pool.append(utils.ThreadWithReturnValue(target=p_train_mnn_thread, args=(audio_path, label_dictionary, chunk)))
        else:
            features = np.empty(shape=(0, N_MFCC, AUDIO_LENGTH, 1))
            thread_pool.append(utils.ThreadWithReturnValue(target=p_train_cnn_thread, args=(audio_path, label_dictionary, chunk)))
    # print a log message for status update
    utils.write_log_msg("TRAIN: creating a total of {0} threads...".format(len(thread_pool)))  
    # start the entire thread pool
    for single_thread in thread_pool:
        single_thread.start()
    # wait for thread pool to return their results of processing
    for single_thread in thread_pool:
        ft, lbl = single_thread.join()
        features = np.vstack([features, ft])
        labels = np.append(labels, lbl)
    # perform final touches to extracted arrays
    features = np.array(features)
    #print(labels)
    labels = np.array(labels, dtype = np.int)

    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # return the extracted features to the calling program
    return features, labels

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   p_train_cnn_thread()                                                                        #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse training audio files in multi-threaded environment for CNN.      #
#                                                                                               #
#***********************************************************************************************#
def p_train_cnn_thread(audio_path, label_dictionary, data):
    # initialize variables
    labels = np.empty(0)
    X = np.empty(shape=(data.shape[0], N_MFCC, AUDIO_LENGTH, 1))
    # process this threads share of workload
    for i in range(data.shape[0]):
        # add a log message to be displayed after processing every 250 files.
        if i % int(CHUNK_SIZE/4) == 0:
            utils.write_log_msg("FEATURE_CNN_TRAIN - {0}".format(i))
        line = data.iloc[i]
        fn = audio_path+line["fname"]
        sound_clip, _ = librosa.core.load(fn, sr=SAMPLE_RATE, res_type='kaiser_fast')
        # Random offset / Padding
        if len(sound_clip) > INPUT_LENGTH:
            max_offset = len(sound_clip) - INPUT_LENGTH
            offset = np.random.randint(max_offset)
            sound_clip = sound_clip[offset:(INPUT_LENGTH+offset)]
        else:
            if INPUT_LENGTH > len(sound_clip):
                max_offset = INPUT_LENGTH - len(sound_clip)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            sound_clip = np.pad(sound_clip, (offset, INPUT_LENGTH - len(sound_clip) - offset), "constant")
        # extract mfcc features
        mfcc = librosa.feature.mfcc(sound_clip, sr = SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = np.expand_dims(mfcc, axis=-1)
        X[i,] = mfcc
        # populate the labels array
        labels = np.append(labels, label_dictionary[line["label"]])
    # return the extracted features to the calling program
    return np.array(X), labels

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   p_predict_cnn_thread()                                                                      #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse prediction audio files in multi-threaded environment for CNN.    #
#                                                                                               #
#***********************************************************************************************#
def p_predict_cnn_thread(audio_path, name_list):
    # initialize variables
    X = np.empty(shape=(len(name_list), N_MFCC, AUDIO_LENGTH, 1))
    # traverse through the name list and process this threads workload
    for i, fname in enumerate(name_list):
        # add a log message to be displayed after processing every 250 files.
        if i % int(CHUNK_SIZE/4) == 0:
            utils.write_log_msg("FEATURE_CNN_PREDICT - {0}...".format(i))
        # read the sound file
        sound_clip,_ = librosa.load(audio_path+fname, sr=SAMPLE_RATE, res_type='kaiser_fast')
        # Random offset / Padding
        if len(sound_clip) > INPUT_LENGTH:
            max_offset = len(sound_clip) - INPUT_LENGTH
            offset = np.random.randint(max_offset)
            sound_clip = sound_clip[offset:(INPUT_LENGTH+offset)]
        else:
            if INPUT_LENGTH > len(sound_clip):
                max_offset = INPUT_LENGTH - len(sound_clip)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            sound_clip = np.pad(sound_clip, (offset, INPUT_LENGTH - len(sound_clip) - offset), "constant")  
        # extract mfcc features
        mfcc = librosa.feature.mfcc(sound_clip, sr = SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = np.expand_dims(mfcc, axis=-1)
        X[i,] = mfcc
    # return the extracted features to the calling program
    return np.array(X)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   read_tr_features()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for reading pre-made features files without any extraction.             #
#                                                                                               #
#***********************************************************************************************#
def read_tr_features():
    # read mnn features
    tr_mnn_features = np.load(TR_FEATURE_NPY.format("mnn"))
    tr_mnn_labels = np.load(TR_LABEL_NPY.format("mnn"))
    # read cnn features
    tr_cnn_features = np.load(TR_FEATURE_NPY.format("cnn"))
    tr_cnn_labels = np.load(TR_LABEL_NPY.format("cnn"))
    # read the dictionary which is same in both cases
    dictionary = json.load(open(LABEL_DICT_NPY))
    # return the read values
    return dictionary, tr_mnn_features, tr_mnn_labels, tr_cnn_features, tr_cnn_labels

def read_tr_features_mnn():
    # read mnn features
    tr_mnn_features = np.load(TR_FEATURE_NPY.format("mnn"))
    tr_mnn_labels = np.load(TR_LABEL_NPY.format("mnn"))
    # read the dictionary which is same in both cases
    dictionary = json.load(open(LABEL_DICT_NPY))
    # return the read values
    return dictionary, tr_mnn_features, tr_mnn_labels

def read_tr_features_cnn():
    # read cnn features
    tr_cnn_features = np.load(TR_FEATURE_NPY.format("cnn"))
    tr_cnn_labels = np.load(TR_LABEL_NPY.format("cnn"))
    # return the read values
    return tr_cnn_features, tr_cnn_labels

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   read_ts_features()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for reading pre-made features files without any extraction.             #
#                                                                                               #
#***********************************************************************************************#
def read_ts_features():
    # read mnn features
    ts_mnn_features = np.load(TS_FEATURE_NPY.format("mnn"))
    ts_mnn_name_list = pickle.load(open(TS_F_NAME_NPY.format("mnn"), "rb"))
    # read cnn features
    ts_cnn_features = np.load(TS_FEATURE_NPY.format("cnn"))
    ts_cnn_name_list = pickle.load(open(TS_F_NAME_NPY.format("cnn"), "rb"))
    # return the read values
    return ts_mnn_features, ts_mnn_name_list, ts_cnn_features, ts_cnn_name_list

def read_ts_features_mnn():
    # read mnn features
    ts_mnn_features = np.load(TS_FEATURE_NPY.format("mnn"))
    ts_mnn_name_list = pickle.load(open(TS_F_NAME_NPY.format("mnn"), "rb"))

    # return the read values
    return ts_mnn_features, ts_mnn_name_list

def read_ts_features_cnn():
    # read cnn features
    ts_cnn_features = np.load(TS_FEATURE_NPY.format("cnn"))
    ts_cnn_name_list = pickle.load(open(TS_F_NAME_NPY.format("cnn"), "rb"))
    # return the read values
    return ts_cnn_features, ts_cnn_name_list

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   store_tr_features()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for storing pre-made features to files.                                 #
#                                                                                               #
#***********************************************************************************************#
def store_tr_features(dictionary, tr_mnn_features, tr_mnn_labels, tr_cnn_features, tr_cnn_labels):
    # save mnn features
    np.save(TR_FEATURE_NPY.format("mnn"), tr_mnn_features)
    np.save(TR_LABEL_NPY.format("mnn"), tr_mnn_labels)
    # save cnn features
    np.save(TR_FEATURE_NPY.format("cnn"), tr_cnn_features)
    np.save(TR_LABEL_NPY.format("cnn"), tr_cnn_labels)
    # save the dictionary which is same in both cases
    json.dump(dictionary, open(LABEL_DICT_NPY,'w'))

def store_tr_features_mnn(dictionary, tr_mnn_features, tr_mnn_labels):
    # save mnn features
    np.save(TR_FEATURE_NPY.format("mnn"), tr_mnn_features)
    np.save(TR_LABEL_NPY.format("mnn"), tr_mnn_labels)
    # save the dictionary which is same in both cases
    json.dump(dictionary, open(LABEL_DICT_NPY,'w'))

def store_tr_features_cnn(tr_cnn_features, tr_cnn_labels):
    # save cnn features
    np.save(TR_FEATURE_NPY.format("cnn"), tr_cnn_features)
    np.save(TR_LABEL_NPY.format("cnn"), tr_cnn_labels)

# ***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   store_ts_features()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for storing pre-made features to files.                                 #
#                                                                                               #
# ***********************************************************************************************#
def store_ts_features(ts_mnn_features, ts_mnn_name_list, ts_cnn_features, ts_cnn_name_list):
    # save mnn features
    np.save(TS_FEATURE_NPY.format("mnn"), ts_mnn_features)
    pickle.dump(ts_mnn_name_list, open(TS_F_NAME_NPY.format("mnn"), "wb"))
    # save cnn features
    np.save(TS_FEATURE_NPY.format("cnn"), ts_cnn_features)
    pickle.dump(ts_cnn_name_list, open(TS_F_NAME_NPY.format("cnn"), "wb"))

def store_ts_features_mnn(ts_mnn_features, ts_mnn_name_list):
    # save mnn features
    np.save(TS_FEATURE_NPY.format("mnn"), ts_mnn_features)
    pickle.dump(ts_mnn_name_list, open(TS_F_NAME_NPY.format("mnn"), "wb"))

def store_ts_features_cnn(ts_cnn_features, ts_cnn_name_list):
    # save cnn features
    np.save(TS_FEATURE_NPY.format("cnn"), ts_cnn_features)
    pickle.dump(ts_cnn_name_list, open(TS_F_NAME_NPY.format("cnn"), "wb"))
