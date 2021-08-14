#!/usr/bin/env python3

#rewritten from fetchOrbit.py
#Cunren Liang, 21-MAR-2018

#update url format in function query_file(). 15-DEC-2020

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


def query_file():

    server = 'https://qc.sentinel1.eo.esa.int/aux_cal/'
    'https://qc.sentinel1.eo.esa.int/aux_cal/?page=1'

    session = requests.Session()
    fileList = []
    for i in range(1, 10000+1):
        #there are 20 products on each page
        query = server + '?page={}'.format(i)
        r = session.get(query, verify=True)
        r.raise_for_status()
        parser = MyHTMLParser('_AUX_CAL_')
        parser.feed(r.text)
        #parser.fileList is a list
        if set(parser.fileList).issubset(set(fileList)):
            break
        else:
            fileList += parser.fileList

    session.close()

    result = []
    for x in fileList:
        mission = x.split('_')[0]
        startingTime = x.split('_')[-2][1:]
        result.append(os.path.join('https://qc.sentinel1.eo.esa.int/product', mission, 'AUX_CAL', startingTime, x+'.SAFE.TGZ'))

    return result


def runCmd(cmd, silent=0):
    import os

    if silent == 0:
        print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='fetch calibration auxiliary data')
    parser.add_argument('-p','--unpack', dest='unpack', type=int, required=True,
            help = 'whether unpack data. 0: yes. 1: no.')

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

    urls = query_file()
    
    print('\ndownloading data')
    downloaded = []
    for x in urls:
        #download data
        x0 = x.split('/')[-1]
        print('downloading: {}'.format(x0))
        success = download_file(x)
        if success is True:
            downloaded.append(x0)
        else:
            print('failed to download URL: {}'.format(x))

    if inps.unpack == 0:
        print('\nunpacking data')
        unpacked = []
        for x in downloaded:
            print('unpacking: {}'.format(x))
            cmd = 'tar -xvzf {} >/dev/null 2>&1'.format(x)
            status = os.system(cmd)
            if status == 0:
                unpacked.append(x)
                os.remove(x)

    print('\ntotal number of files downloaded: {}'.format(len(downloaded)))
    if len(downloaded) == len(urls):
        print('all files downloaded')
    else:
        print('number of files not successfully downloaded: {}'.format(len(urls)-len(downloaded)))

    if inps.unpack == 0:
        print('\ntotal number of files unpacked: {}'.format(len(unpacked)))
        if len(unpacked) == len(downloaded):
            print('all downloaded files unpacked')
        else:
            print('number of files not successfully unpacked: {}\n'.format(len(downloaded)-len(unpacked)))







