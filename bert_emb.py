"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import simplejson as json
import numpy as np
import _pickle as pickle
import gzip
import math

from bert import tokenization
import tensorflow as tf

from service.client import BertClient
bc = BertClient()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_integer("batch_size", None, 128)
flags.DEFINE_integer("bs_per_shard", None, 100)

def main(argv):
    tokenizer = tokenization.FullTokenizer(vocab_file="../uncased_L-24_H-1024_A-16/vocab.txt")
    
    data = open(FLAGS.input_file, 'r').read().strip().split('\n')
    print ('data size:', len(data))

    ft = open(FLAGS.output_file + '_tokens.txt', 'w')
    
    raw_batch = []
    tok_batch = []
    shard = []
    bs = 0
    shard_no = 0
    for i in range(len(data)):
        raw_batch.append(data[i])
        tok_batch.append(['[CLS]'] + tokenizer.tokenize(data[i]) + ['[SEP]'])
        if len(raw_batch) == FLAGS.batch_size:
            res = bc.encode(raw_batch)
            '''
            for b in range(FLAGS.batch_size):
                if len(res[b]) != len(tok_batch[b]):
                    print (len(res[b]), len(tok_batch[b]))
            '''
            embs = []
            for r in range(len(res)):
                embs.append(np.asarray(res[r][:len(tok_batch[r])], 'float16'))
            shard += embs
            ft.write('\n'.join([' '.join(tok) for tok in tok_batch]) + '\n')
            raw_batch = []
            tok_batch = []
            bs += 1
            print (bs)
        if bs == FLAGS.bs_per_shard:
            print ('shard size', len(shard))
            fe = gzip.GzipFile(FLAGS.output_file+'_%d.gzip.pkl'%shard_no, 'wb')
            pickle.dump(shard, fe, -1)
            #np.save(FLAGS.output_file+'_%d.npy'%shard_no, shard)
            fe.close()
            shard = []
            bs = 0
            shard_no += 1

    if len(shard) > 0:
        fe = gzip.GzipFile(FLAGS.output_file+'_%d.gzip.pkl'%(shard_no), 'wb')
        pickle.dump(shard, fe, -1)
        #np.save(FLAGS.output_file+'_%d.npy'%(shard_no+1), shard)
        fe.close()

            
if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("bs_per_shard")
  tf.app.run()
