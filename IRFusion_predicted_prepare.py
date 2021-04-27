#!/usr/bin/env python
import sys
import os

feature_input = sys.argv[1]
output_path = sys.argv[2]
genome = sys.argv[3]

prefix = os.path.split(feature_input)[1].split('.')[0]
if not os.path.isdir(output_path):
    os.makedirs(output_path)

def write(filename, data):
    with open(filename, 'w') as fp:
        for line in data:
            fp.write('\t'.join(map(str, line))+'\n')

introns = []
features = []
intron_read_counts = []

feature_start, feature_end = 1, 163 # the start and end position of feature
intron_read_counts_position = 155
with open(feature_input, 'r') as fp:
    first_line = True
    for line in fp:
        if first_line:
            first_line = False
            continue
        tab = line.strip().split('\t')
        chro, e_start, i_start, i_end, e_end, strand = tab[0].split('|')[:6]
        key = tab[0]
        introns.append([chro, i_start, i_end, key, 0, strand])
        features.append(tab[feature_start:feature_end+1])
        intron_read_counts.append(tab[intron_read_counts_position])

introns_file = os.path.join(output_path, prefix+'_sequence_intron.bed')
features_file = os.path.join(output_path, prefix+'_sequence_feature.txt')
write(introns_file, introns)
write(features_file, features)

intron_read_counts_file = os.path.join(output_path, prefix+'_sequence_intron_read_counts.txt')
with open(intron_read_counts_file, 'w') as fp:
        for line in intron_read_counts:
            fp.write(line+'\n')

sequences_5 = []
sequences_3 = []
length_seq = 400
len_one = length_seq // 2
with open(introns_file, 'r') as fp:
    for line in fp:
        chro, start, end, key, score, strand = line.strip().split('\t')[:6]
        start = int(start)
        end = int(end)
        # get 5'
        s_5 = start - len_one # get len_one bp
        e_5 = start + len_one + 1 # get len_one bp
        seq_5 = [chro, s_5, e_5, key, score, strand]
        # get 3'
        s_3 = end - len_one - 1 # get len_one bp
        e_3 = end + len_one # get len_one bp
        seq_3 = [chro, s_3, e_3, key, score, strand]
        # strand map
        seq_5_strand = seq_5 if strand == '+' else seq_3
        seq_3_strand = seq_3 if strand == '+' else seq_5
        sequences_5.append(seq_5_strand)
        sequences_3.append(seq_3_strand)

introns_5_bed = os.path.join(output_path, prefix+'_sequence_intron_5.bed')
introns_3_bed = os.path.join(output_path, prefix+'_sequence_intron_3.bed')
write(introns_5_bed, sequences_5)
write(introns_3_bed, sequences_3)

introns_5_file = os.path.join(output_path, prefix+'_sequence_intron_5.fa')
introns_3_file = os.path.join(output_path, prefix+'_sequence_intron_3.fa')
command = 'bedtools getfasta '+\
            ' -fi '+genome+\
            ' -bed '+introns_5_bed+\
            ' -s '+\
            ' -name '+\
            ' -fo '+introns_5_file
os.system(command)
command = 'bedtools getfasta '+\
            ' -fi '+genome+\
            ' -bed '+introns_3_bed+\
            ' -s '+\
            ' -name '+\
            ' -fo '+introns_3_file
os.system(command)
