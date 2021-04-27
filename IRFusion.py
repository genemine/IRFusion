#!/usr/bin/python
# This is the command line entry point to predict IR event.
# coded at: Central South University, Changsha 410083, P.R. China
# coded by: Zhenpeng Wu.
# contact: zhenpeng@csu.edu.cn
# Jan. 14, 2021

import os
import sys
import optparse

prompt_information = 'IRFusion.py [-m build] [-g <genome.fa>] [-a <annotation.gtf>] [-i <index_EIE_path>]'\
    +'\n'+'IRFusion.py [-m predict] [-i <index_EIE_path>] [-b <predicted.bam>] [-o <output_path>] [-n <number_of_threads>]'
optParser = optparse.OptionParser(
    usage = prompt_information,
    )

if os.path.exists('/proc/cpuinfo'): # linux
    all_threads = int(os.popen('cat /proc/cpuinfo| grep "processor"| wc -l').read().strip())
else: # mac, not support windows
    all_threads = int(os.popen('sysctl -n machdep.cpu.thread_count').read().strip())
threads = all_threads // 2 if all_threads > 2 else 1

# mode:
# first build EIE model index, -m build;
# then predict IR evnet, -m predict
optParser.add_option('-m', '--mode', action='store', type='string', dest='mode_flag', help='build: build EIE model index, extract: extract intron feature.')
optParser.add_option('-g', '--genome_file', action='store', type='string', dest='genome_file', help='genome fasta file of Ensembl.') # genome
optParser.add_option('-a', '--annotation_file', action='store', type='string', dest='annotation_file', help='an annotation file in Ensembl GTF format.') # annotation in GTF format file
optParser.add_option('-b', '--bam_file', action='store', type='string', dest='bam_file', help='extracted bam file.') # bam
optParser.add_option('-o', '--output_path', action='store', type='string', dest='output_path', help='output path.') # output
optParser.add_option('-i', '--index_EIE_path', action='store', type='string', dest='EIE_path', help='the path of EIE idnex.') # EIE index
optParser.add_option('-n', '--number_of_threads', action='store', type='int', default=threads, dest='threads', help='If user not specificed, default 1/2 of all threads of machine')
optParser.add_option('-c', '--number_of_read_counts', action='store', type='int', default=0, dest='intron_read_counts', help='IR event can set an additional condition (i.e. read count of aligning intron is intron_read_counts at least). If user not specificed, default 0')

(opts, args) = optParser.parse_args()

def get_abspath(rel_path,tmp_name='get_abspath_tmp.txt'):
    rel_path_file = rel_path+tmp_name
    os.system('echo "This file for produce abpath" > '+rel_path_file)
    abspath = os.path.abspath(rel_path_file).replace(tmp_name,'')
    os.system('rm '+rel_path_file)
    return abspath

def mkdir_function(path):
    if path != None and not os.path.isdir(path):
        os.makedirs(path)

### handle path of output
mkdir_function(opts.output_path)
if opts.output_path != None and opts.output_path.split('/')[-1] != '': # output path
    opts.output_path =  opts.output_path+'/'
if opts.output_path != None and os.path.isabs(opts.output_path) == False: # get abspath
    opts.output_path = get_abspath(opts.output_path)

# handle path of EIE
mkdir_function(opts.EIE_path)
if opts.EIE_path != None and opts.EIE_path.split('/')[-1] != '':
    opts.EIE_path =  opts.EIE_path+'/'
if opts.EIE_path != None and os.path.isabs(opts.EIE_path) == False: # get abspath
    opts.EIE_path = get_abspath(opts.EIE_path)

#######      #######
####### main #######
#######      #######
if opts.mode_flag == 'build':
    build_EIE = ' IRTools.py -m build '+' -g '+opts.genome_file+' -a '+opts.annotation_file+' -i '+opts.EIE_path+' -p on '
    os.system(build_EIE)
    cp_genome = 'cp '+opts.genome_file+' '+os.path.join(opts.EIE_path,'genome.fa')
    os.system(cp_genome)
elif opts.mode_flag == 'predict':
    extract_intron_feature = ' IRTools.py -m extract '+' -i '+opts.EIE_path+' -b '+opts.bam_file+' -o '+opts.output_path+' -n '+str(opts.threads)
    os.system(extract_intron_feature)
    feature_input = os.path.join(opts.output_path, 'feature_intron.txt')
    genome = os.path.join(opts.EIE_path, 'genome.fa')
    exe_file = sys.argv[0].split('/')[-1] # current execute file
    exe_path = sys.argv[0].replace(exe_file,'') # get current execute path
    model_weights_file = os.path.join(exe_path, 'IRFusion_model/fusion_feature_model_weight.h5')
    predict_IR = ' IRFusion_predicted.py '+' '+feature_input+' '+opts.output_path+' '+genome+' '+model_weights_file+' '+str(opts.intron_read_counts)
    os.system(predict_IR)
else:
    print(prompt_information)
    print('Please check the model!')
