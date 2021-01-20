import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

K_RUNS = 4

def _get_clip_seq(df, subject_list,args):
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):

            if i_class==0: # split test-retest into 4
                seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(seq)
                    y.append(label_seq)
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))

                X.append(seq)
                y.append(label_seq)
                
    X_len = tf.convert_to_tensor([len(seq) for seq in X])

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, padding="post",
        dtype='float'
    )

    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, padding="post",
        dtype='float'
    )
    y = np.array([array[0] for array in y])
    
    return tf.convert_to_tensor(X_padded,dtype='float32'), X_len ,tf.convert_to_tensor(y_padded,dtype='float32')


def _get_bhv_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
        in {0, 1, ..} if args.mode=='class'
        in R if args.mode=='reg'
    c: clip seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    # optional arguments
    d = vars(args)

    # regression or classification
    if 'mode' not in d:
        args.mode = 'class'
    if args.mode=='class':
        label = 'y'
    elif args.mode=='reg':
        label = args.bhv

    # permutation test
    if 'shuffle' not in d:
        args.shuffle = False
    if args.shuffle:
        # different shuffle for each iteration
        np.random.seed(args.i_seed)
        # get scores for all participants without bhv_df
        train_label = df[(df['Subject'].isin(subject_list)) &
            (df['c']==1) & (df['timepoint']==0)][label].values
        np.random.shuffle(train_label) # inplace

    k_clip = len(np.unique(df['c']))
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    c = []

    for ii, subject in enumerate(subject_list):
        for i_clip in range(k_clip):

            if i_clip==0: #handle test retest differently
                seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)][features].values
                if args.shuffle:
                    label_seqs = np.ones(seqs.shape[0])*train_label[ii]
                else:
                    label_seqs = df[(df['Subject']==subject) & 
                        (df['c'] == 0)][label].values
                clip_seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)]['c'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    clip_seq = clip_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(tf.convert_to_tensor(seq,dtype='float32'))
                    if args.mode=='class':
                        y.append(tf.convert_to_tensor(label_seq,dtype=np.int32))
                    elif args.mode=='reg':
                        y.append(tf.convert_to_tensor(label_seq,dtype='float32'))
                    c.append(tf.convert_to_tensor(clip_seq,dtype=np.int32))
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)][features].values
                if args.shuffle:
                    label_seq = np.ones(seq.shape[0])*train_label[ii]
                else:
                    label_seq = df[(df['Subject']==subject) & 
                        (df['c'] == i_clip)][label].values
                clip_seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)]['c'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))

                X.append(tf.convert_to_tensor(seq,dtype='float32'))
                if args.mode=='class':
                    y.append(tf.convert_to_tensor(label_seq,dtype=np.int32))
                elif args.mode=='reg':
                    y.append(tf.convert_to_tensor(label_seq,dtype='float32'))
                c.append(tf.convert_to_tensor(clip_seq,dtype=np.int32))

    X_len = tf.convert_to_tensor([len(seq) for seq in X],dtype=np.int32)

    # pad sequences
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post",value=0.,dtype='float32')
    if args.mode == 'class':
        y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post",value=0.,dtype='int32')
    elif args.mode == 'reg':
        y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post",value=0.,dtype='float32')
    c = tf.keras.preprocessing.sequence.pad_sequences(c, padding="post",value=0)

    return X, X_len, y, c