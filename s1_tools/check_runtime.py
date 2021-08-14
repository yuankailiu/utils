#!/usr/bin/env python3
# This code should be run after the processing of topsApp 
# Can generate a runtime bar plot for all pairs
# Required: 
#	- will read a log file `cmd_runall.log` generated from `runallstep.sh`
#	- need to be run in the working dir that contains all pairs dirs, e.g., `process`
# This code can be useful for:
#	- checking the general runtime of processing a track
#	- identifying potential issues with anomalously short runtime (short scene due to lack of slice)
#	- simply cool to make a plot like this!
# YKL @ May 2021

import os
import glob
import argparse
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Plot the runtime bar chart for pairs of topsApp processed interferograms \n \
        - workflow:\n \
            1. go into each pair directory \n \
            2. read the `cmd_runall.log` log file and read the runtime for each step \n \
            3. make the plot and save it \n'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', dest='outname', type=str, default='topsRuntime.png',
            help = 'output fig filename. Default: `topsRuntime.png`')
    parser.add_argument('-t', dest='title', type=str, default=' ',
            help = 'title of the plot, e.g., Descending_021')
    parser.add_argument('-s', dest='figsize', type=float, nargs='+', default=[20, 6],
            help = 'figure size, default: [20, 6]')

    return parser.parse_args()



if __name__ == '__main__':
    
    # read the user inputs
    inps = cmdLineParse()
    outname = inps.outname
    title   = inps.title + '(TopsApp Processing)'

    # get path and logfile name
    logfile = 'cmd_runall.log'
    workDir = os.getcwd()
    os.chdir(workDir)
    print('Read {} under each pair dir, under the current working path: {}'.format(logfile, workDir))
    folders = sorted(glob.glob('*-*/'))

    # total step of standard topsApp processing: from startup to geocode
    Nstep = 21

    # loop over all the pair dirs
    pairs = []
    pairs_time = []
    for i in range(len(folders)):
        folder = folders[i]
        os.chdir(folder)
        msg = folder[:-1]
        if not os.path.exists(logfile):
            os.chdir(workDir)
            msg += '\tnoLogFile'
            continue
        with open(logfile,"r") as fi:
            tmp = []
            for ln in fi:
                if ln.startswith("real	"):
                    tstr = ln.split()[1]
                    min = int(tstr.split('m')[0])
                    sec = float(tstr.split('m')[1][:-1])
                    secs = min*60 + sec
                    tmp.append(secs)
        if len(tmp) == Nstep:
            pairs.append(folder[:-1])
            pairs_time.append(np.array(tmp))
            msg += '\telapse={:.2f} m'.format(np.sum(np.array(tmp))/60)
        else:
            msg += '\t< 21steps'
        os.chdir(workDir)
        print(msg)

    pairs_time = np.array(pairs_time).T
    print('Num of pairs processed:',len(pairs))

    # step names
    steps = [
    'startup',
    'preprocess',
    'computeBaselines',
    'verifyDEM',
    'topo',
    'subsetoverlaps',
    'coarseoffsets',
    'coarseresamp',
    'overlapifg',
    'prepesd',
    'esd',
    'rangecoreg',
    'fineoffsets',
    'fineresamp',
    'ion',
    'burstifg',
    'mergebursts',
    'filter',
    'unwrap',
    'unwrap2stage',
    'geocode'
    ]

    # plotting
    Pos    = np.arange(len(pairs))
    plt.figure(figsize=inps.figsize)
    for i in range(Nstep):
        plt.bar(Pos, pairs_time[i,:], bottom=np.sum(pairs_time[0:i,:], axis=0),
                width=1.0, edgecolor='w', linewidth=0.2, label=steps[i])
    plt.xticks(Pos, pairs, rotation=90)
    plt.xlim(0,None)
    plt.yticks(np.arange(0,25200,1800), np.arange(0,25200,1800)/3600)
    plt.ylabel('Hours')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig('./{}'.format(outname), dpi=300, bbox_inches='tight')

    # ending message
    print('Plotting is completed, saved to ./{}'.format(outname))