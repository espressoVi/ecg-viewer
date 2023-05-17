#!/usr/bin/env python
import os
import numpy as np
import logging
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.io import loadmat
from utils import bandpass_filter, notch_filter
from scipy.signal import decimate, resample
import matplotlib.pyplot as plt
NUM_CLASSES = 27

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

NUM_CHANNEL = 12
DATA_DIR = './data/'
DIAG_DICT = './classes.json'
PLOT_DIR = './plots'

class Example:
    def __init__(self, fid: str, diag, data, age, sex, sample_Fs: int, fold: int ):
        self.id = fid
        self.diag = diag
        self.data = data
        self.age = age
        self.sex = sex
        self.sample_Fs = sample_Fs
        self.fold = fold

class ECGPipeline:
    def __init__(self, input_directory):
        self.input_directory = input_directory
        self.diag_dict_dir = DIAG_DICT 
        self.num_classes = NUM_CLASSES - 3 #for equivalent classes

    def load_file(self,filename):
        x = loadmat(filename)
        data = np.asarray(x['val'], dtype=np.float32)
        new_file = filename.replace('.mat','.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file,'r') as f:
            header_data=f.readlines()
        return data, header_data

    def _load_diag_dict(self):
        with open(self.diag_dict_dir, 'r') as f:
            d = json.load(f)
        return d['code_to_class']

    def create_example(self, all_data):
        data, header = all_data
        tmp = header[0].split(' ')
        fid = tmp[0].split('.')[0]
        gain = int(header[1].split(' ')[2].split('/')[0])
        num_channels = int(tmp[1])
        sample_Fs = int(tmp[2])
        num_samples = int(tmp[3])
        assert num_samples == data.shape[1]
        assert gain == 1000
        assert num_channels == data.shape[0] == NUM_CHANNEL
        datetime = tmp[4]+' '+tmp[5]
        for f in header[1:]:
            if 'Dx' in f:
                diagnosis = f[5:-1].split(',')
            elif 'Age' in f:
                try:
                    age = int(f[6:-1])
                except:
                    age = None
            elif 'Sex' in f:
                sex = f[6:-1]
            else:
                continue
        code_to_class = self._load_diag_dict()
        diag = np.zeros(self.num_classes)
        for dx in diagnosis: 
            if dx in code_to_class:
                diag[code_to_class[dx]] = 1

        return Example(fid = fid,
                    diag = diag,
                    data = data,
                    age = age,
                    sex = sex,
                    sample_Fs = sample_Fs,
                    fold = 0,)

class Feature:
    def __init__(self, data, fid, diag = None, fold = None, repeat = None):
        self.data = data
        self.diag = diag
        self.fid = fid
        self.fold = fold
        self.repeat = repeat

class ECGExtractor:
    def __init__(self, sample_num = 4096, sample_Fs = 500, lowcut = 0.001, highcut = 15.0,):
        self.input_dir = DATA_DIR 
        self.pipeline = ECGPipeline(self.input_dir)
        self.num_channels = 12
        self.sample_num = sample_num
        self.overlap = 512
        self.sample_Fs = sample_Fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notchcut = [50,60]
        self.max_instances = 1

    def get_feature(self, filename):
        one_file = self.pipeline.load_file(filename)
        example = self.pipeline.create_example(one_file)
        if example is None:
            return []
        data = example.data
        if data.shape[1] < self.sample_num:
            data = np.pad(data, ((0,0),(0,self.sample_num - data.shape[1])))
        num_sample = data.shape[1]
        resampler = self.Resample(example.sample_Fs, num_sample)
        slices = resampler(data)
        for i in slices:
            assert i.shape == (self.num_channels, self.sample_num)
        features = []
        if self.lowcut and self.highcut:
            slices = [bandpass_filter(slice_data, self.lowcut, self.highcut, self.sample_Fs) for slice_data in slices]
        if self.notchcut:
            slices = [notch_filter(slice_data, self.notchcut, self.sample_Fs) for slice_data in slices]

        for cnt,slice_data in enumerate(slices):
            feature = Feature(data = slice_data/1000,
                              diag = example.diag,
                              fid = example.id+ '_'+str(cnt),
                              fold = example.fold,
                              repeat = True if cnt > 0 else False)
            features.append(feature)
        return features[:self.max_instances]

    def normalize_channels(self,data):
        mean = np.mean(data)
        std = np.std(data)
        assert std != 0
        return (data-mean)/std

    def Resample(self,sample_Fs, length):
        cut_length = int(self.sample_num*(sample_Fs/self.sample_Fs))
        overlap_length =  int(self.overlap*(sample_Fs/self.sample_Fs))
        idx = [(i,i+cut_length) for i in range(0,length, cut_length-overlap_length) if i+cut_length < length]
        idx = idx[:self.max_instances]
        def resample_500(data):
            return [data[:,ix[0]:ix[1]] for ix in idx]
        def resample_1000(data):
            temp = [data[:,ix[0]:ix[1]] for ix in idx]
            return [decimate(data,int(sample_Fs/self.sample_Fs), axis = 1) for data in temp]
        def resample_257(data):
            temp = [data[:,ix[0]:ix[1]] for ix in idx]
            return [resample(data,self.sample_num, axis = 1) for data in temp]
        if sample_Fs == 500:
            return resample_500
        elif sample_Fs == 1000:
            return resample_1000
        elif sample_Fs == 257:
            return resample_257
        else:
            raise ValueError('Not recognized frequency')

ecg = ECGExtractor()
lead_names = {0:'I', 1:'II', 2:'III',3:'aVR',4:'aVL',5:'aVF',6:'V1',7:'V2',8:'V3',9:'V4',10:'V5',11:'V6'}
lead_map = {0:0, 1:3, 2:6, 3:9, 4:1, 5:4, 6: 7, 7:10, 8:2, 9:5, 10:8, 11:11}
disease = {0: 'IAVB', 1: 'AF', 2: 'AFL', 3: 'Brady', 4: 'CRBBB',
        5: 'IRBBB', 6: 'LAnFB', 7: 'LAD', 8: 'LBBB', 9: 'LQRSV',
        10: 'NSIVCB', 11: 'PR', 12: 'PAC', 13: 'PVC', 14: 'LPR',
        15: 'LQT', 16: 'QAb', 17: 'RAD', 18: 'SA',
        19: 'SB', 20: 'NSR', 21: 'STach', 22: 'TAb', 23: 'TInv', }

def getFile(filename):
    file = os.path.join(DATA_DIR, filename + '.mat')
    feature = ecg.get_feature(file)
    return feature[0].data, feature[0].diag

def diagStr(diag):
    st = ''
    for i in np.where(diag==1)[0]:
        st = st + disease[i]+ ' ,'
    return st

def plot(filename):
    data, diag = getFile(filename)
    X = np.linspace(0,8.192, 4096)
    major_xticks = np.arange(0,8.192,0.2)
    minor_xticks = np.arange(0,8.192,0.04)
    major_yticks = np.arange(-10,10,0.5)
    minor_yticks = np.arange(-10,10,0.1)
    fig = plt.figure(figsize = (12,8), dpi = 300, )
    gs = fig.add_gridspec(4,4)
    minim,maxim = np.amin(data), np.amax(data)
    for i in range(12):
        idx = lead_map[i]
        ax = fig.add_subplot(gs[i//4,i%4])
        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor = True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor = True)
        ax.set_frame_on(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        mean = np.mean(data[idx])
        dat = data[idx]-mean
        ax.set_ylim(minim - 0.001,maxim+0.001)
        ax.set_xlim(0,8.192)
        ax.grid(which = 'minor', color = '#E83E42', alpha = 0.2)
        ax.grid(which = 'major', color = '#E83E42', alpha = 0.5)
        ax.set_title(lead_names[idx],loc = 'left',y = 0.85, fontsize='small')
        ax.plot(X, dat, color = '#000000', alpha = 0.6, linewidth = 0.6)
    ax = fig.add_subplot(gs[3,:])
    ax.set_title(lead_names[0],loc = 'left',y = 0.6, fontsize='small')
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor = True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor = True)
    ax.set_frame_on(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    mean = np.mean(data[idx])
    dat = data[0]-mean
    ax.set_ylim(np.amin(dat)-0.1,np.amax(dat)+0.1)
    ax.set_xlim(0,8.192)
    ax.grid(which = 'minor', color = '#E83E42', alpha = 0.2)
    ax.grid(which = 'major', color = '#E83E42', alpha = 0.5)
    ax.plot(X, dat, color = '#000000', alpha = 0.6, )

    fig.suptitle(filename + ' | Diagnosis : ' + diagStr(diag))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    plt.savefig(os.path.join(PLOT_DIR, filename + '.png'), bbox_inches = 'tight')
    plt.close(fig)

if __name__ == '__main__':
    for i,f in enumerate(os.listdir('./data')):
        nam = f.split('.')[0]
        plot(nam)
        print(f'File : {nam} | {i}/{len(os.listdir("./data"))}')
