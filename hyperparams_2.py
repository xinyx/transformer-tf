# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/content4_300.txt'
    target_train = 'corpora/title4_20.txt'
    source_val = 'corpora/val_content4_300.txt'
    target_val = 'corpora/val_title4_20.txt'
    source_test = 'corpora/final_test_content4_300.txt'
    src_vocab = 'preprocessed/content.vocab.tsv'
    tgt_vocab = 'preprocessed/title.vocab.tsv'
    
    # training
    num_gpus = 4
    batch_size = 25 # alias = N
    lr = 0.001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    moving_average_decay = 0.99
    val_every_batchs = 10000
    summary_every_steps = 100
    
    # model
    x_max_len = 301 # include </S>
    y_max_len = 21  # include </S>

    # embedding
    src_vocab_size = 40000
    tgt_vocab_size = 40000
    fused_vocab = 'preprocessed/fused_vocab.tsv'
    emb_size = 300
    glove_path = "corpora/glove.840B.300d.txt"

    hidden_units = 768 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.2
    sinusoid = True # If True, use sinusoid. If false, positional embedding.
   
    # features
    case_size = 4
    pos_size = 20
    bert_size = 1024
    case_path = '../transformer/corpora/case2emb.pkl'
    pos_path = '../transformer/corpora/pos2emb.pkl'
    '''
    train_pos_path = 'corpora/train_content_pos4_nodup_300.txt'
    val_pos_path = 'corpora/val_content_pos4_nodup_300.txt'
    test_pos_path = 'corpora/test_content_pos4_nodup_300.txt'
    train_case_path = 'corpora/train_content_case4_nodup_300.txt'
    val_case_path = 'corpora/val_content_case4_nodup_300.txt'
    test_case_path = 'corpora/test_content_case4_nodup_300.txt'
    train_bert_path = 'corpora/glove_bert/content4_300_bert_%d.gzip.pkl'
    val_bert_path = 'corpora/glove_bert/val_content4_300_bert_%d.gzip.pkl'
    test_bert_path = 'corpora/glove_bert/test_content4_300_bert_%d.gzip.pkl'
    '''

    train_bert_file_num = 47
    val_bert_file_num = 1
    test_bert_file_num = 1

    # beam search
    beam_size = 1
    max_len = 10 
    end_id = y_max_len-1
    
