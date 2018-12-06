# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams_2 import Hyperparams as hp
from data_load_2 import *
from modules import *
import os, codecs
from tqdm import tqdm
from functools import reduce
from operator import mul
import commands

def get_num_params():    
    num_params = 0   
    for variable in tf.trainable_variables():   
        shape = variable.get_shape()     
        num_params += reduce(mul, [dim.value for dim in shape], 1)     
    return num_params

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        self.is_training = is_training
        def get_loss(x, y, decoder_inputs, scope, case=None, pos=None, bert=None, reuse_variables=None):
            with tf.variable_scope('get_loss', reuse=reuse_variables):
                self.glove_lookup_table = tf.get_variable('others_lookup_table',
                        dtype=tf.float32,
                        initializer=init_embedding)
                # Fix PAD embedding to zeros
                self.glove_lookup_table = tf.concat([tf.zeros(shape=(1, hp.emb_size)), 
                                                    self.glove_lookup_table[1:, :]], 0)

                # Encoder
                with tf.variable_scope("encoder"):
                    ## Embedding
                    self.enc = embedding(x, 
                                          vocab_size=len(w2i), 
                                          num_units=hp.emb_size, 
                                          lookup_table=self.glove_lookup_table,
                                          scale=True,
                                          scope="enc_embed")
                    
                    ## Masks
                    self.enc_masks = tf.sign(tf.abs(tf.reduce_sum(self.enc, axis=-1))) # (?, x_max_len)

                    ## Case embedding
                    if hp.case_size > 0:
                        self.case_lookup_table = tf.get_variable('case_lookup_table',
                                dtype=tf.float32,
                                initializer=case_embedding)

                        self.enc = tf.concat([self.enc, 
                            tf.nn.embedding_lookup(self.case_lookup_table, case)], -1)

                    ## Pos embedding
                    if hp.pos_size > 0:
                        self.pos_lookup_table = tf.get_variable('pos_lookup_table',
                                dtype=tf.float32,
                                initializer=pos_embedding)
                        self.enc = tf.concat([self.enc,
                            tf.nn.embedding_lookup(self.pos_lookup_table, pos)], -1)

                    ## Bert embedding
                    if hp.bert_size > 0:
                        self.enc = tf.concat([self.enc, bert], -1)

                    ## Positional Encoding
                    if hp.sinusoid:
                        self.enc += positional_encoding(x,
                                          num_units=hp.emb_size + hp.case_size + hp.pos_size + hp.bert_size, 
                                          zero_pad=False, 
                                          scale=False,
                                          scope="enc_pe")
                    else:
                        self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                                          vocab_size=hp.x_max_len, 
                                          num_units=hp.emb_size + hp.case_size + hp.pos_size + hp.bert_size, 
                                          zero_pad=False, 
                                          scale=False,
                                          scope="enc_pe")

                    ## Dropout
                    self.enc = tf.layers.dropout(self.enc, 
                                                rate=hp.dropout_rate, 
                                                training=tf.convert_to_tensor(is_training))
                    
                    ## Fit embedding 
                    self.enc = tf.layers.dense(self.enc, hp.hidden_units, activation=tf.nn.relu)

                    ## Blocks
                    for i in range(hp.num_blocks):
                        with tf.variable_scope("num_blocks_{}".format(i)):
                            ### Multihead Attention
                            self.enc = multihead_attention(queries=self.enc, 
                                                            keys=self.enc, 
                                                            query_masks=self.enc_masks,
                                                            key_masks=self.enc_masks,
                                                            num_units=hp.hidden_units, 
                                                            num_heads=hp.num_heads, 
                                                            dropout_rate=hp.dropout_rate,
                                                            is_training=is_training,
                                                            reuse=reuse_variables,
                                                            causality=False)
                            
                            ### Feed Forward
                            self.enc = feedforward(self.enc, 
                                                   num_units=[4*hp.hidden_units, hp.hidden_units],
                                                   reuse=reuse_variables)
                
                # Decoder
                with tf.variable_scope("decoder"):
                    ## Embedding
                    self.dec = embedding(decoder_inputs, 
                                          vocab_size=len(w2i), 
                                          num_units=hp.emb_size,
                                          lookup_table=self.glove_lookup_table,
                                          scale=True, 
                                          scope="dec_embed")
                    self.dec_masks = tf.sign(tf.abs(tf.reduce_sum(self.dec, axis=-1))) # (?, y_max_len)
                    
                    ## Positional Encoding
                    if hp.sinusoid:
                        self.dec += positional_encoding(decoder_inputs,
                                          num_units=hp.emb_size, 
                                          zero_pad=False, 
                                          scale=False,
                                          scope="dec_pe")
                    else:
                        self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0), [tf.shape(decoder_inputs)[0], 1]),
                                          vocab_size=hp.y_max_len, 
                                          num_units=hp.emb_size,
                                          zero_pad=False, 
                                          scale=False,
                                          scope="dec_pe")
                    
                    ## Dropout
                    self.dec = tf.layers.dropout(self.dec, 
                                                rate=hp.dropout_rate, 
                                                training=tf.convert_to_tensor(is_training))
                   
                    ## Fit embedding 
                    self.dec = tf.layers.dense(self.dec, hp.hidden_units, activation=tf.nn.relu)

                    ## Blocks
                    for i in range(hp.num_blocks):
                        with tf.variable_scope("num_blocks_{}".format(i)):
                            ## Multihead Attention ( self-attention)
                            self.dec = multihead_attention(queries=self.dec, 
                                                            keys=self.dec,
                                                            query_masks=self.dec_masks,
                                                            key_masks=self.dec_masks,
                                                            num_units=hp.hidden_units, 
                                                            num_heads=hp.num_heads, 
                                                            dropout_rate=hp.dropout_rate,
                                                            is_training=is_training,
                                                            reuse=reuse_variables,
                                                            causality=True, 
                                                            scope="self_attention")
                            
                            ## Multihead Attention ( vanilla attention)
                            self.dec = multihead_attention(queries=self.dec, 
                                                            keys=self.enc,
                                                            query_masks=self.dec_masks,
                                                            key_masks=self.enc_masks,
                                                            num_units=hp.hidden_units, 
                                                            num_heads=hp.num_heads,
                                                            dropout_rate=hp.dropout_rate,
                                                            is_training=is_training, 
                                                            reuse=reuse_variables,
                                                            causality=False,
                                                            scope="vanilla_attention")
                            
                            ## Feed Forward
                            self.dec = feedforward(self.dec, 
                                                   num_units=[4*hp.hidden_units, hp.hidden_units],
                                                   reuse=reuse_variables)
                    
                # Final linear projection
                self.logits = tf.layers.dense(self.dec, len(w2i))
                self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
                self.istarget = tf.to_float(tf.not_equal(y, 0))
                self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, y))*self.istarget)/ (tf.reduce_sum(self.istarget))
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(y, depth=len(w2i)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

            return self.mean_loss, self.acc, self.preds

        def average_gradients(tower_grads):    
            with tf.variable_scope('average_grad'):    
                average_grads = []   
                # 枚举所有的变量和变量在不同GPU上计算得出的梯度。     
                for grad_and_vars in zip(*tower_grads): 
                    # 计算所有GPU上的梯度平均值。       
                    grads = []  
                    for i, (g, _) in enumerate(grad_and_vars):
                        if g == None:   # for multi-gpu, g is None except for gpu:0
                            #print ('None grad for var', i)
                            continue
                        expanded_g = tf.expand_dims(g, 0)   
                        grads.append(expanded_g)  
                    grad = tf.concat(grads, 0) 
                    grad = tf.reduce_mean(grad, 0) 

                    v = grad_and_vars[0][1]    
                    grad_and_var = (grad, v)     
                    # 将变量和它的平均梯度对应起来。  
                    average_grads.append(grad_and_var)   
                # 返回所有变量的平均梯度，这个将被用于变量的更新。        
                return average_grads
            
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('inputs'):
                self.xs, self.ys, self.cases, self.poss, self.berts = get_train_tfrecords_batch_data() 
                self.val_xs, self.val_ys, self.val_cases, self.val_poss, self.val_berts = get_val_tfrecords_batch_data()  
                self.test_xs, self.test_cases, self.test_poss, self.test_berts = get_test_tfrecords_batch_data()  
                self.xs = tf.cast(self.xs, tf.int32) 
                self.ys = tf.cast(self.ys, tf.int32) 
                self.cases = tf.cast(self.cases, tf.int32)
                self.poss = tf.cast(self.poss, tf.int32)
                self.berts = tf.cast(self.berts, tf.float32)

                self.val_xs = tf.cast(self.val_xs, tf.int32) 
                self.val_ys = tf.cast(self.val_ys, tf.int32) 
                self.val_cases = tf.cast(self.val_cases, tf.int32)
                self.val_poss = tf.cast(self.val_poss, tf.int32)
                self.val_berts = tf.cast(self.val_berts, tf.float32)
                
                self.test_xs = tf.cast(self.test_xs, tf.int32) 
                self.test_cases = tf.cast(self.test_cases, tf.int32)
                self.test_poss = tf.cast(self.test_poss, tf.int32)
                self.test_berts = tf.cast(self.test_berts, tf.float32)

                # define decoder inputs 
                self.decoder_inputs = tf.concat((tf.ones_like(self.ys[:, :1])*2, self.ys[:, :-1]), -1) # 2:<S>

            # Training Scheme
            self.opt = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            with tf.variable_scope('multi_GPU_loss'):
                 self.tower_grads = []  
                 self.tower_losses = [] 
                 self.tower_preds = []  
                 self.tower_accs = []  
                 reuse_variables = False
                 for gpuid in range(4):
                     xs_i = self.xs[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]
                     ys_i = self.ys[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]
                     decoder_inputs_i = self.decoder_inputs[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]
                     cases_i = self.cases[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]
                     poss_i = self.poss[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]
                     berts_i = self.berts[gpuid*hp.batch_size:(gpuid+1)*hp.batch_size]

                     with tf.device('/gpu:%d' % gpuid):
                         with tf.name_scope('GPU_%d' % gpuid) as scope:
                             cur_loss_, acc_, preds_ = get_loss(xs_i, ys_i, decoder_inputs_i, 
                                                                scope, 
                                                                cases_i,  
                                                                poss_i,
                                                                berts_i,
                                                                reuse_variables)
                             reuse_variables = True
                             cur_grads_ = self.opt.compute_gradients(cur_loss_, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                             self.tower_grads.append(cur_grads_)
                             self.tower_losses.append(cur_loss_)
                             self.tower_preds.append(preds_)
                             self.tower_accs.append(acc_)
                             
                 self.tower_preds = tf.concat(self.tower_preds, 0)
                 self.tower_accs = tf.reduce_mean(self.tower_accs, 0)

            self.avg_tower_loss = tf.reduce_mean(self.tower_losses, 0)
            self.avg_grads = average_gradients(self.tower_grads)
            self.apply_gradient_op = self.opt.apply_gradients(self.avg_grads, global_step=self.global_step)
               
            # Summary 
            tf.summary.scalar('lookup_table_0', tf.reduce_sum(self.glove_lookup_table[0], -1))
            tf.summary.histogram('glove_lookup_table', self.glove_lookup_table)
            tf.summary.scalar('acc', self.tower_accs)
            tf.summary.scalar('avg_loss', self.avg_tower_loss)
            tf.summary.scalar('lr', self.opt._lr)
            
            ## masks
            tf.summary.text('enc_masks', tf.as_string(tf.reduce_sum(self.enc_masks, -1)))
            
            ## grads
            for g,v in self.avg_grads:
                tf.summary.histogram('grads/%s'%v.op.name, g)


                    
            self.summary_op = tf.summary.merge_all()

def validate(g, sess, mname):
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/" + mname, "w", "utf-8") as fout:
        list_of_refs, hypotheses = [], [] 
        #for i in range(len(X) // (hp.batch_size * hp.num_gpus)):  
        num_steps = 2000//hp.batch_size//hp.num_gpus
        for i in range(num_steps):
            ### Get mini-batches
            xs_batch, ys_batch, cases_batch, poss_batch, berts_batch = sess.run([g.val_xs, g.val_ys, g.val_cases, g.val_poss, g.val_berts])
            sources = [' '.join([i2w[idx] for idx in line]) for line in xs_batch]
            sources = [src.split('</S>')[0] for src in sources]
            targets = [' '.join([i2w[idx] for idx in line]) for line in ys_batch]
            targets = [tgt.split('</S>')[0] for tgt in targets]

            ### Autoregressive inference
            preds = np.zeros((hp.batch_size*hp.num_gpus, hp.y_max_len), np.int32)
            for j in range(hp.y_max_len):         
                preds_ = sess.run(g.tower_preds, {g.xs: xs_batch, 
                                                  g.ys: preds,
                                                  g.cases: cases_batch,
                                                  g.poss: poss_batch,
                                                  g.berts: berts_batch})  
                preds[:, j] = preds_[:, j]

            ### Write to file
            for source, target, pred in zip(sources, targets, preds): # sentence-wise 
                got = " ".join(i2w[idx] for idx in pred).split("</S>")[0].strip()    
                fout.write("- source: " + source +"\n") 
                fout.write("- expected: " + target + "\n")  
                fout.write("- got: " + got + "\n\n") 
                fout.flush()  
                list_of_refs.append(target)
                hypotheses.append(got)

        ## Calculate rouge score
        # score = corpus_bleu(list_of_refs, hypotheses)
        codecs.open('.refs', 'w', 'utf-8').write('\n'.join(list_of_refs).strip())
        codecs.open('.hyps', 'w', 'utf-8').write('\n'.join(hypotheses).strip())
        status, output = commands.getstatusoutput('rouge -f ".hyps" ".refs" --avg')
        fout.write("Rouge Score = " + output)

def inference(g, sess, mname):
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open("results/" + mname, "w", "utf-8") as fout:
        num_steps = 800//hp.batch_size//hp.num_gpus
        for i in range(num_steps):
            ### Get mini-batches
            xs_batch, cases_batch, poss_batch, berts_batch = sess.run([g.test_xs, g.test_cases, g.test_poss, g.test_berts])
            sources = [' '.join([i2w[idx] for idx in line]) for line in xs_batch]
            sources = [src.split('</S>')[0] for src in sources]

            ### Autoregressive inference
            preds = np.zeros((hp.batch_size*hp.num_gpus, hp.y_max_len), np.int32)
            for j in range(hp.y_max_len):         
                preds_ = sess.run(g.tower_preds, {g.xs: xs_batch, 
                                                  g.ys: preds,
                                                  g.cases: cases_batch,
                                                  g.poss: poss_batch,
                                                  g.berts: berts_batch})  
                preds[:, j] = preds_[:, j]

            ### Write to file
            for source, pred in zip(sources, preds): # sentence-wise 
                got = " ".join(i2w[idx] for idx in pred).split("</S>")[0].strip()    
                fout.write("- source: " + source +"\n") 
                fout.write("- got: " + got + "\n\n") 
                fout.flush()  

def test_tfrecords(g, sess):
    val_xs, val_ys = sess.run([g.val_xs, g.val_ys])
    print (' '.join([i2w[i] for i in val_xs[0]]))
    print (' '.join([i2w[i] for i in val_ys[0]]))

if __name__ == '__main__':                
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    fw = open('preds.txt', 'w')
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             summary_op=None,
                             ready_op=None,
                             save_model_secs=0)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess, g.graph.as_default():
        print ('all variables', len(tf.global_variables()))
        print ('trainable variables', len(tf.trainable_variables()))
        print ('params size', get_num_params())

        with open('trainable_variables.tsv', 'w') as fw:     
            for var in tf.trainable_variables():    
                fw.write(var.op.name + '\t' + var.device + '\t' + ','.join(map(str, var.shape.as_list())) + '\n')

        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break

            # each epoch runs g.num_batch batchs
            num_batch = 10000  
            val_every_steps = hp.val_every_batchs // hp.num_gpus

            for step in tqdm(range(num_batch/hp.num_gpus), total=num_batch/hp.num_gpus, ncols=70, leave=False, unit='b'):
                gs = sess.run(g.global_step)   

                if gs % hp.summary_every_steps == 0:
                    _, summary, srcs, tgts, preds = sess.run([g.apply_gradient_op, 
                                           g.summary_op, 
                                           g.xs[:5],
                                           g.ys[:5],
                                           g.tower_preds[:5]])
                    sv.summary_computed(sess, summary)
                    ## preds
                    for n in range(5):
                        fw.write('src - ' + ' '.join([i2w[idx] for idx in srcs[n]]) + '\n')
                        fw.write('tgt - ' + ' '.join([i2w[idx] for idx in tgts[n]]) + '\n')
                        fw.write('got - ' + ' '.join([i2w[idx] for idx in preds[n]]) + '\n\n')

                else:
                    sess.run(g.apply_gradient_op)

                if gs != 0 and gs % val_every_steps == 0:
                    sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                    print ('Validating...')
                    validate(g, sess, 'model_epoch_%02d_gs_%d' % (epoch, gs))
                    print ('Inferencing...')
                    inference(g, sess, 'inference_epoch_%02d_gs_%d' % (epoch, gs))
    
    fw.close()
    print("Done")    
    

