#!/usr/bin/env python
import sys
import os
import numpy as np
from IRFusion_model import predict_model

feature_input = sys.argv[1]
output_path = sys.argv[2]
genome = sys.argv[3]
model_weights_file = sys.argv[4]
intron_read_counts_threshold = int(sys.argv[5])

if not os.path.isdir(output_path):
    os.makedirs(output_path)
    
output_path_tmp = os.path.join(output_path, 'tmp')
os.system('IRFusion_predicted_prepare.py '+feature_input+' '+output_path_tmp+' '+genome)

prefix = os.path.split(feature_input)[1].split('.')[0]
features_file = os.path.join(output_path_tmp, prefix+'_sequence_feature.txt')
introns_5_file = os.path.join(output_path_tmp, prefix+'_sequence_intron_5.fa')
introns_3_file = os.path.join(output_path_tmp, prefix+'_sequence_intron_3.fa')

def read_file(filename):
    data_file = []
    with open(filename, 'r') as fp:
        for line in fp:
            data_file.append(line.strip())
    return data_file

bases = ['A', 'C', 'G', 'T']
length = 400
length_seq = length+1
length_bases = len(bases)

def encoded(seq):
    """
    :param seq:
    :return the seq one hot encoded:
    """
    x = np.zeros( (length_seq, length_bases) )
    for i, val in enumerate(seq):
        if val in bases:
            x[i, bases.index(val)] = 1
    return x

feature_data = read_file(features_file)
introns_5_data = read_file(introns_5_file)
introns_3_data = read_file(introns_3_file)

seq_5, seq_3, feature = [], [], []
feature_num = len(feature_data[0].split('\t'))
for i, f in enumerate(feature_data):
    seq = introns_5_data[i*2+1]
    x = encoded(seq).reshape(1, length_seq, length_bases)
    seq_5 += [x]
    seq = introns_3_data[i*2+1]
    x = encoded(seq).reshape(1, length_seq, length_bases)
    seq_3 += [x]
    temp_f = np.zeros( (feature_num, ) )
    f = f.split('\t')
    for i, val in enumerate(f):
        temp_f[i] = val
    temp_f = temp_f.reshape(1, feature_num, )
    feature += [temp_f]
seq_5, seq_3, feature = np.vstack(seq_5), np.vstack(seq_3), np.vstack(feature)
# print('All data shape -> seq_5 {} seq_3 {} feature {}'.format(seq_5.shape, seq_3.shape, feature.shape))

print('Predicting intron retention events')
model = predict_model()
model.load_weights(model_weights_file, by_name=True)
score = model.predict([seq_5, seq_3, feature], verbose=1)

def write(filename, data):
    with open(filename, 'w') as fp:
        for line in data:
            fp.write('\t'.join(map(str, line))+'\n')

intron_read_counts_file = os.path.join(output_path_tmp, prefix+'_sequence_intron_read_counts.txt')
intron_read_counts_data = read_file(intron_read_counts_file)
result = []
first_line_content = ['chro', 'start', 'end', 'intron_id', 'IR_score', 'intron_read_counts', 'label']
result.append(first_line_content)
for i, f in enumerate(feature_data):
    intron_id = introns_5_data[i*2].split(':')[0].split('>')[1]
    chro, e_start, i_start, i_end, e_end, strand, gene_id = intron_id.split('|')[:7]
    i_r_c = int(intron_read_counts_data[i])
    if score[i][0] >= 0.5 and i_r_c >= intron_read_counts_threshold:
        label = 1
    else:
        label = 0
    intron_id = '-'.join([chro, e_start, i_start, i_end, e_end, gene_id])
    data = [chro, i_start, i_end, intron_id, score[i][0], i_r_c, label]
    result.append(data)
result_file = os.path.join(output_path, 'IRFusion_IR_predict.txt')
write(result_file, result)

os.system('rm -r '+output_path_tmp)
print('Predicting intron retention events complete!')
