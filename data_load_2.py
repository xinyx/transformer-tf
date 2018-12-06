# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams_2 import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import gzip
import os
import platform
if platform.python_version().split('.')[0] == '2':
    import cPickle as pickle
else:
    import _pickle as pickle


def load_fs_emb():
    if os.path.exists('.case_embedding.pkl') and os.path.exists('.pos_embedding.pkl'):
        print ('loading case and pos embedding...')
        case_embedding = pickle.load(open('.case_embedding.pkl', 'rb'))
        pos_embedding = pickle.load(open('.pos_embedding.pkl', 'rb'))
        return case_embedding, pos_embedding

    print ('building case and pos embedding...')
    case2emb = pickle.load(open(hp.case_path, 'rb'))
    pos2emb = pickle.load(open(hp.pos_path, 'rb'))
    case2id, pos2id = {}, {}
    for id_,case in enumerate(case2emb.keys()):
        case2id[case] = id_
    for id_,pos in enumerate(pos2emb.keys()):
        pos2id[pos] = id_
    
    ## the lase embedding is for <PAD>
    case_embedding = np.zeros((len(case2emb)+1, hp.case_size))
    for case,id_ in case2id.items():
        case_embedding[id_] = case2emb[case]

    pos_embedding = np.zeros((len(pos2emb)+1, hp.pos_size))
    for pos,id_ in pos2id.items():
        pos_embedding[id_] = pos2emb[pos]

    case_embedding = np.asarray(case_embedding, np.float32)
    pos_embedding = np.asarray(pos_embedding, np.float32)
    pickle.dump(case_embedding, open('.case_embedding.pkl', 'wb'), 1)
    pickle.dump(pos_embedding, open('.pos_embedding.pkl', 'wb'), 1)
    return case_embedding, pos_embedding


def load_vocab():
    if os.path.exists('.word2idx.pkl') and os.path.exists('.idx2word.pkl') and os.path.exists('.embedding.pkl'):
        print ('loading vocab and initialized embeddings...')
        word2idx = pickle.load(open('.word2idx.pkl', 'rb'))
        idx2word = pickle.load(open('.idx2word.pkl', 'rb'))
        embedding = pickle.load(open('.embedding.pkl', 'rb'))
        return word2idx, idx2word, embedding
        #return word2idx, idx2word

    print ('building vocab and initializing embeddings...')
    src_vocab = [line.split()[0] for line in codecs.open(hp.src_vocab, 'r', 'utf-8').read().strip().split('\n')][:hp.src_vocab_size]
    tgt_vocab = [line.split()[0] for line in codecs.open(hp.tgt_vocab, 'r', 'utf-8').read().strip().split('\n')][:hp.tgt_vocab_size]
    reserved_tokens = src_vocab[:4] #<PAD> <UNK> <S> </S>
    vocab = reserved_tokens + list(set(src_vocab[4:] + tgt_vocab[4:]))
    print ('vocab size', len(vocab))
    
    with codecs.open(hp.fused_vocab, 'w', 'utf-8') as fw:
        fw.write('\n'.join(vocab))

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    glove_data = open(hp.glove_path, 'r').read().strip().split('\n')
    glove_size = len(glove_data[0].split()) - 1
    glove_embedding_weights = np.empty((len(glove_data), glove_size))
    glove_w2e = {}
    for i,line in enumerate(glove_data):
        glove_embedding_weights[i, :] = map(float, line.split()[1:])
        glove_w2e[line.split()[0]] = map(float, line.split()[1:])
    
    scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
    embedding = np.random.uniform(low=-scale, high=scale, size=(len(word2idx), hp.emb_size))
    nFound = 0
    for word,idx in word2idx.items():
        if word in glove_w2e:
            nFound += 1
            embedding[idx, :min(hp.emb_size, glove_size)] = glove_w2e[word][:min(hp.emb_size, glove_size)]

    embedding = np.asarray(embedding, np.float32)
    print ('vocab found in glove', nFound)

    pickle.dump(word2idx, open('.word2idx.pkl', 'wb'), 1)
    pickle.dump(idx2word, open('.idx2word.pkl', 'wb'), 1)
    pickle.dump(embedding, open('.embedding.pkl', 'wb'), 1)
    return word2idx, idx2word, embedding
    #return word2idx, idx2word
    

def create_data(source_sents, target_sents, x_max_len=hp.x_max_len, y_max_len=hp.y_max_len): 

    if target_sents == None:
        target_sents = ['' for i in range(len(source_sents))]
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        source_sent = ' '.join(source_sent.split(' ')[:x_max_len-1])
        target_sent = ' '.join(target_sent.split(' ')[:y_max_len-1])
        x = [w2i.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [w2i.get(word, 1) for word in (target_sent + u" </S>").split()] 
        x_list.append(np.array(x))
        y_list.append(np.array(y))
        Sources.append(source_sent)
        Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), x_max_len], np.int32)
    Y = np.zeros([len(y_list), y_max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, x_max_len-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, y_max_len-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    src_sents = codecs.open(hp.source_train, 'r', 'utf-8').read().strip().split("\n")
    tgt_sents = codecs.open(hp.target_train, 'r', 'utf-8').read().strip().split("\n")
    
    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Y

def load_val_data():
    src_sents = codecs.open(hp.source_val, 'r', 'utf-8').read().strip().split("\n")
    tgt_sents = codecs.open(hp.target_val, 'r', 'utf-8').read().strip().split("\n")
    
    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Sources, Targets

def get_tensor_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size*hp.num_gpus, 
                                capacity=hp.batch_size*hp.num_gpus*64,   
                                min_after_dequeue=hp.batch_size*hp.num_gpus*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

# src max len: 301, tgt max len: 21
def prepare_tfrecords():
    train_content = codecs.open(hp.source_train, 'r', 'utf-8').read().strip().split('\n')
    train_title = codecs.open(hp.target_train, 'r', 'utf-8').read().strip().split('\n')
    val_content = codecs.open(hp.source_val, 'r', 'utf-8').read().strip().split('\n')
    val_title = codecs.open(hp.target_val, 'r', 'utf-8').read().strip().split('\n')
    test_content = codecs.open(hp.source_test, 'r', 'utf-8').read().strip().split('\n')

    train_content_case = codecs.open(hp.train_case_path, 'r', 'utf-8').read().strip().split('\n')
    val_content_case = codecs.open(hp.val_case_path, 'r', 'utf-8').read().strip().split('\n')
    test_content_case = codecs.open(hp.test_case_path, 'r', 'utf-8').read().strip().split('\n')
    train_content_pos = codecs.open(hp.train_pos_path, 'r', 'utf-8').read().strip().split('\n')
    val_content_pos = codecs.open(hp.val_pos_path, 'r', 'utf-8').read().strip().split('\n')
    test_content_pos = codecs.open(hp.test_pos_path, 'r', 'utf-8').read().strip().split('\n')



    case2emb = pickle.load(open(hp.case_path, 'rb'), encoding='latin1')
    pos2emb = pickle.load(open(hp.pos_path, 'rb'), encoding='latin1')
    case2id, pos2id = {}, {}
    for id_,case in enumerate(case2emb.keys()):
        case2id[case] = id_

    for id_,pos in enumerate(pos2emb.keys()):
        pos2id[pos] = id_

    src_ids, _, _, _ = create_data(test_content, None, 301, 21)
    test_tfr = tf.python_io.TFRecordWriter('corpora/test.tfrecords')
    fp = gzip.GzipFile(hp.test_bert_path%0, 'rb')
    berts = pickle.load(fp) # (?, seq_len, 1024)
    fp.close()
    print ('test file 0, size %d' % (len(berts)))
    for j in range(len(berts)):
        LEN = min(300, len(berts[j]))
        if len(test_content[j].split()) != len(test_content_case[j].split()):
            print (j)
        if len(test_content_case[j].split()) != len(test_content_pos[j].split()):
            print (j)

        case_ids, pos_ids = [], []
        for case in test_content_case[j].split()[:LEN]:
            case_ids.append(case2id[case])
        case_ids += [len(case2id)]*(301-len(case_ids))
        for pos in test_content_pos[j].split()[:LEN]:
            pos_ids.append(pos2id[pos])
        pos_ids += [len(pos2id)]*(301-len(pos_ids))

        # pad berts
        bert_pad = berts[j][:300]
        bert_pad = np.concatenate([bert_pad, np.zeros((301, 1024))], 0)[:301]   # 301*1024
        bert_pad = bert_pad.reshape(-1)


        example = tf.train.Example(features=tf.train.Features(
            feature = {
                    'src': tf.train.Feature(int64_list = tf.train.Int64List(value=list(src_ids[j]))),
                    'case': tf.train.Feature(int64_list = tf.train.Int64List(value=case_ids)),
                    'pos': tf.train.Feature(int64_list = tf.train.Int64List(value=pos_ids)),
                    'bert': tf.train.Feature(float_list = tf.train.FloatList(value=bert_pad))
                }))
        test_tfr.write(example.SerializeToString())

    print ('done!')
    test_tfr.close()

    src_ids, tgt_ids, _, _ = create_data(val_content, val_title, 301, 21)
    val_tfr = tf.python_io.TFRecordWriter('corpora/val.tfrecords')
    fp = gzip.GzipFile(hp.val_bert_path%0, 'rb')
    berts = pickle.load(fp) # (?, seq_len, 1024)
    fp.close()
    print ('val file 0, size %d' % (len(berts)))
    for j in range(len(berts)):
        LEN = min(300, len(berts[j]))
        if len(val_content[j].split()) != len(val_content_case[j].split()):
            print (j)
        if len(val_content_case[j].split()) != len(val_content_pos[j].split()):
            print (j)

        case_ids, pos_ids = [], []
        for case in val_content_case[j].split()[:LEN]:
            case_ids.append(case2id[case])
        case_ids += [len(case2id)]*(301-len(case_ids))
        for pos in val_content_pos[j].split()[:LEN]:
            pos_ids.append(pos2id[pos])
        pos_ids += [len(pos2id)]*(301-len(pos_ids))

        # pad berts
        bert_pad = berts[j][:300]
        bert_pad = np.concatenate([bert_pad, np.zeros((301, 1024))], 0)[:301]   # 301*1024
        bert_pad = bert_pad.reshape(-1)


        example = tf.train.Example(features=tf.train.Features(
            feature = {
                    'src': tf.train.Feature(int64_list = tf.train.Int64List(value=list(src_ids[j]))),
                    'tgt': tf.train.Feature(int64_list = tf.train.Int64List(value=list(tgt_ids[j]))),
                    'case': tf.train.Feature(int64_list = tf.train.Int64List(value=case_ids)),
                    'pos': tf.train.Feature(int64_list = tf.train.Int64List(value=pos_ids)),
                    'bert': tf.train.Feature(float_list = tf.train.FloatList(value=bert_pad))
                }))
        val_tfr.write(example.SerializeToString())

    print ('done!')
    val_tfr.close()

    src_ids, tgt_ids, _, _ = create_data(train_content, train_title, 301, 21)
    for i in range(9, hp.train_bert_file_num):
        idx = 12800*i
        train_tfr = tf.python_io.TFRecordWriter('corpora/train_%d.tfrecords' % i)
        fp = gzip.GzipFile(hp.train_bert_path % i, 'rb')
        berts = pickle.load(fp) # (?, seq_len, 1024)
        fp.close()
        print ('train file %d, size %d' % (i, len(berts)))
        for j in range(len(berts)):
            LEN = min(300, len(berts[j]))
            if len(train_content[idx+j].split()) != len(train_content_case[idx+j].split()):
                print (j)
            if len(train_content_case[idx+j].split()) != len(train_content_pos[idx+j].split()):
                print (j)

            case_ids, pos_ids = [], []
            for case in train_content_case[idx+j].split()[:LEN]:
                case_ids.append(case2id[case])
            case_ids += [len(case2id)]*(301-len(case_ids))
            for pos in train_content_pos[idx+j].split()[:LEN]:
                pos_ids.append(pos2id[pos])
            pos_ids += [len(pos2id)]*(301-len(pos_ids))

            # pad berts
            bert_pad = berts[j][:300]
            bert_pad = np.concatenate([bert_pad, np.zeros((301, 1024))], 0)[:301]   # 301*1024
            bert_pad = bert_pad.reshape(-1)


            example = tf.train.Example(features=tf.train.Features(
                feature = {
                        'src': tf.train.Feature(int64_list = tf.train.Int64List(value=list(src_ids[idx+j]))),
                        'tgt': tf.train.Feature(int64_list = tf.train.Int64List(value=list(tgt_ids[idx+j]))),
                        'case': tf.train.Feature(int64_list = tf.train.Int64List(value=case_ids)),
                        'pos': tf.train.Feature(int64_list = tf.train.Int64List(value=pos_ids)),
                        'bert': tf.train.Feature(float_list = tf.train.FloatList(value=bert_pad))
                    }))
            train_tfr.write(example.SerializeToString())

        print (idx, 'done!')
    train_tfr.close()

def get_train_tfrecords_batch_data():
    filenames = ['../transformer/corpora/train_%d.tfrecords'%i for i in range(hp.train_bert_file_num)]
    #filenames = ['corpora/val.tfrecords']

    # setting num_epochs to None will infinitly loop the data
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
           features={
               'src': tf.FixedLenFeature([301], tf.int64),
               'tgt': tf.FixedLenFeature([21], tf.int64),
               'case': tf.FixedLenFeature([301], tf.int64),
               'pos': tf.FixedLenFeature([301], tf.int64),
               'bert': tf.FixedLenFeature([301,1024], tf.float32)})
    src,tgt,case,pos,bert = features['src'],features['tgt'],features['case'],features['pos'],features['bert']

    srcs, tgts, cases, poss, berts = tf.train.batch(
            [src, tgt, case, pos, bert],
            num_threads = 8,
            batch_size = hp.batch_size*hp.num_gpus,
            capacity = hp.batch_size*hp.num_gpus*32,
            allow_smaller_final_batch=False)

    return srcs, tgts, cases, poss, berts

def get_val_tfrecords_batch_data():
    filenames = ['../transformer/corpora/val.tfrecords']

    # setting num_epochs to None will infinitly loop the data
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
           features={
               'src': tf.FixedLenFeature([301], tf.int64),
               'tgt': tf.FixedLenFeature([21], tf.int64),
               'case': tf.FixedLenFeature([301], tf.int64),
               'pos': tf.FixedLenFeature([301], tf.int64),
               'bert': tf.FixedLenFeature([301,1024], tf.float32)})
    src,tgt,case,pos,bert = features['src'],features['tgt'],features['case'],features['pos'],features['bert']

    srcs, tgts, cases, poss, berts = tf.train.batch(
            [src, tgt, case, pos, bert],
            num_threads = 2,
            batch_size = hp.batch_size*hp.num_gpus,
            capacity = hp.batch_size*hp.num_gpus*8,
            allow_smaller_final_batch=True)

    return srcs, tgts, cases, poss, berts

def get_test_tfrecords_batch_data():
    filenames = ['../transformer/corpora/final_test.tfrecords']

    # setting num_epochs to None will infinitly loop the data
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
           features={
               'src': tf.FixedLenFeature([301], tf.int64),
               #'tgt': tf.FixedLenFeature([21], tf.int64),
               'case': tf.FixedLenFeature([301], tf.int64),
               'pos': tf.FixedLenFeature([301], tf.int64),
               'bert': tf.FixedLenFeature([301,1024], tf.float32)})
    src,case,pos,bert = features['src'],features['case'],features['pos'],features['bert']

    srcs, cases, poss, berts = tf.train.batch(
            [src, case, pos, bert],
            num_threads = 2,
            batch_size = hp.batch_size*hp.num_gpus,
            capacity = hp.batch_size*hp.num_gpus*8,
            allow_smaller_final_batch=True)

    return srcs, cases, poss, berts

def test_runner(sess, srcs):
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    i = 0
    while (True):
        if coord.should_stop():
            try:
                _ = sess.run(srcs)
                i += 1
            except tf.errors.OutOfRangeError:
                print ('read done', i)
                break
            finally:
                coord.request_stop()
        else:
            print ('should stop', i)
            break
    coord.request_stop()
    coord.join(threads)

w2i, i2w, init_embedding = load_vocab()
case_embedding, pos_embedding = load_fs_emb()
#w2i, i2w = load_vocab()
#prepare_tfrecords()
