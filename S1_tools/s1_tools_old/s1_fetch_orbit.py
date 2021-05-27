#!/usr/bin/env python3

#rewritten from fetchOrbit.py
#Cunren Liang, 21-MAR-2018
#update: use new downloading URL of ESA orbit
#        a little change to query format
#        Cunren Liang, 04-JUL-2018

import os
import sys
import glob
import argparse
import datetime
import requests
from html.parser import HTMLParser
import numpy as np


class MyHTMLParser(HTMLParser):

    def __init__(self, keyword):
        HTMLParser.__init__(self)
        self.fileList = []
        self.in_td = False
        self.in_a = False
        self.keyword = keyword

    def handle_starttag(self, tag, attrs):
        if tag == 'td':
            self.in_td = True
        elif tag == 'a':
            self.in_a = True

    def handle_data(self,data):
        if self.in_td and self.in_a:
            if self.keyword in data:
                self.fileList.append(data.strip())

    def handle_tag(self, tag):
        if tag == 'td':
            self.in_td = False
            self.in_a = False
        elif tag == 'a':
            self.in_a = False


def download_file(url, outdir='.'):
    '''
    Download file to specified directory.
    '''

    session = requests.session()

    path = os.path.join(outdir, os.path.basename(url))
    #print('Downloading URL: ', url)
    request = session.get(url, stream=True, verify=True)

    try:
        val = request.raise_for_status()
        success = True
    except:
        success = False

    if success:
        with open(path,'wb') as f:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
    session.close()

    return success


def query_file(zipname, otype='precise'):

    server = 'https://qc.sentinel1.eo.esa.int/'
    server_download = 'http://aux.sentinel1.eo.esa.int/'

    datefmt = "%Y%m%dT%H%M%S"

    fields = zipname.split('_')
    sat = fields[-10].split('/')[-1]
    tbef_slc = datetime.datetime.strptime(fields[-5], datefmt)
    taft_slc = datetime.datetime.strptime(fields[-4], datefmt)

    if otype == 'precise':
        delta = datetime.timedelta(days=2)
        url = server + 'aux_poeorb'
        url_download = server_download + 'POEORB'
    elif otype == 'restituted':
        #delta = datetime.timedelta(hours=6)
        delta = datetime.timedelta(days=2)
        url = server + 'aux_resorb'
        url_download = server_download + 'RESORB'

    queryfmt = "%Y-%m-%d"
    timebef = (tbef_slc - delta).strftime(queryfmt)
    timeaft = (taft_slc + delta).strftime(queryfmt)
    #query = url + '/?validity_start_time={0}..{1}'.format(timebef, timeaft)
    
    session = requests.Session()
    fileList = []
    for i in range(1, 10000+1):
        #there are 20 products on each page
        #query = url + '/?mission={}&validity_start_time={}..{}&page={}'.format(sat, timebef, timeaft, i)
        query = url + '/?sentinel1__mission={}&validity_start={}..{}&page={}'.format(sat, timebef, timeaft, i)

        r = session.get(query, verify=True)
        r.raise_for_status()
        parser = MyHTMLParser('_OPER_AUX_')
        parser.feed(r.text)
        #parser.fileList is a list
        if set(parser.fileList).issubset(set(fileList)):
            break
        else:
            fileList += parser.fileList

    result = None
    for filename in fileList:
        fields = filename.split('_') 
        taft = datetime.datetime.strptime(fields[-1][0:15], datefmt)
        tbef = datetime.datetime.strptime(fields[-2][1:16], datefmt)
        
        #time extension at both ends: 10 s, 1/3 of a slice
        #change to 1 hr
        time_extension = 3600
        if (tbef <= tbef_slc-datetime.timedelta(seconds=time_extension)) and \
           (taft >= taft_slc+datetime.timedelta(seconds=time_extension)):

            #result = os.path.join(url, filename)
            process_date = filename.split('_')[-3]
            process_year=process_date[0:4]
            process_month=process_date[4:6]
            process_day=process_date[6:8]
            result = os.path.join(url_download, process_year, process_month, process_day, filename) + '.EOF'

            break

    session.close()

    return result


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='fetch orbits')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='directory which has the slc zip files. only zip files recognized.')
    parser.add_argument('-t','--otype', dest='otype', type=int, default=0,
            help = 'type of orbit to download. 0: precise (default). 1: restituted. 2: try precise first, then restituted.')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    safenames = sorted(glob.glob(os.path.join(inps.input, 'S1*_IW_SLC_*.zip')))
    nsafe = len(safenames)

    if nsafe == 0:
        print('no SLC zip files found, exit...')
        sys.exit(0)
    # else:
    #     print('downloading orbit for the following files:')
    #     print('==============================================')
    #     for x in safenames:
    #         print((x.split('/'))[-1])
    #     print('==============================================\n')

    #orbit files downloaded
    orbitnames = []
    #safe files with orbit not successfully downloaded
    safenames2 = []
    print('')
    for i in range(nsafe):
        safename0 = (safenames[i].split('/'))[-1]
        print('downloading orbit file for {}'.format(safename0))
        
        #get url
        if inps.otype in [0, 2]:
            otype = 'precise'
        else:
            otype = 'restituted'
        url = query_file(safenames[i], otype=otype)
    
        if url == None and inps.otype == 2:
            url = query_file(safenames[i], otype='restituted')

        if url == None:
            print('there is no orbit file for this SLC, skip...')
            safenames2.append(safename0)
            continue

        #download orbit
        orbitname = (url.split('/'))[-1]
        if orbitname not in orbitnames:
            #download data
            success = download_file(url)
            if success is True:
                orbitnames.append(orbitname)
            else:
                safenames2.append(safename0)
                print('failed to download URL: {}'.format(url))
    
    print('\ntotal number of orbit files downloaded: {}'.format(len(orbitnames)))
    if safenames2 != []:
        print('no orbit was downloaded for the following files:')
        for x in safenames2:
            print(x)
    else:
        print('orbits for all SLCs are downloaded\n')



