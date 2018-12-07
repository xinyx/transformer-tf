import _pickle as pickle
import gzip
import numpy as np
import argparse
from bert import tokenization

parser = argparse.ArgumentParser()
parser.add_argument("--raw_file", type=str, required=True)
parser.add_argument("--tkn_file", type=str, required=True)
parser.add_argument("--bert_file", type=str, required=True)
parser.add_argument("--bert_file_num", type=int, required=True)
parser.add_argument("--bert_file_size", type=int, required=True)
parser.add_argument("--save_file", type=str, required=True)
#parser.add_argument("--align_file", type=str, required=True)

args = parser.parse_args()

tokenizer = tokenization.FullTokenizer(vocab_file="uncased_L-24_H-1024_A-16/vocab.txt")

def align(raw):
    aln_idx = [1]
    aln_line = ['[CLS]']
    for word in raw:
        tks = tokenizer.tokenize(word)
        aln_line += tks
        idx = aln_idx[-1] + len(tks)
        if idx > 348:
            aln_idx.append(349)
            break
        else:
            aln_idx.append(idx)
    aln_line = aln_line[:349] + ['[SEP]']
    return aln_idx, aln_line

def main():
    raw = open(args.raw_file, 'r').read().strip().split('\n')
    tkn = open(args.tkn_file, 'r').read().strip().split('\n')
    #aln = open(args.align_file, 'w')
    for i in range(args.bert_file_num):
        print ('loading...', i)
        fr = gzip.GzipFile(args.bert_file%i, 'rb')
        bert = pickle.load(fr)
        fr.close()
        print ('bert size', len(bert))
        shard = []
        idx_base = i*args.bert_file_size
        for j in range(len(bert)):
            raw_line = raw[idx_base+j].split()
            tkn_line = tkn[idx_base+j].split()

            raw_bert = bert[j]
            save_bert = []
            align_idx, align_line = align(raw_line)
            if tkn_line[:349] != align_line[:349]:
                print (tkn_line)
                print (align_line)
            if len(bert[j]) != len(align_line):
                print (len(bert[j]), len(align_line))
            assert(align_idx[-1]+1 == len(align_line))

            for k in range(len(align_idx)-1):
                # token not in raw line
                if (align_idx[k] == align_idx[k+1]):
                    save_bert.append(np.zeros(1024, 'float16'))
                else:
                    save_bert.append(np.mean(raw_bert[align_idx[k]:align_idx[k+1]], 0))
            save_bert = np.asarray(save_bert, 'float16')
            shard.append(save_bert)
        print ('save aligned bert...')
        fw = gzip.GzipFile(args.save_file%i, 'wb')
        pickle.dump(shard, fw, -1)
        fw.close()
        

if __name__ == "__main__":
    main()
