import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from scipy.io.wavfile import read
from librosa.core import resample

import os
import random
LEN = 2 # sample 2 sec clip
EPS = 1e-8

def get_files():

    # Domain A training and testing files
    train_files = os.listdir('/home/nitya/projects/gpu_projects/amogh/Data/clean')
    random.shuffle(train_files)
    train_A_files = train_files[:40]
    test_A_files = []
    train_B_files = train_files[40:80]
    test_B_files = train_files[80:]
        
    return train_A_files, train_B_files, test_A_files, test_B_files

# Get noise files used to generate mixtures
def get_all_noise_files(dataset='noise_new'):
    if dataset == 'noise_new':
        ambience_files = os.listdir('/home/nitya/projects/gpu_projects/amogh/Data/noise')
        random.shuffle(ambience_files)
        # files = {}
        # files[0] = ambience_files
    return ambience_files

def get_noise_files(all_noise_files,noise_class_ids):
    noise_files = []
    for c in noise_class_ids:
        noise_files += all_noise_files[c]
    random.shuffle(noise_files)
    return noise_files, noise_files

import soundfile as sf
# Dataset for custom noises
class DapsNoise(data.Dataset):
    def __init__(self,clean_files,noise_files,sr,clip_samples,pure_noise,snr,flag):
        self.clean_root_dir = '/home/nitya/projects/gpu_projects/amogh/Data/clean'
        self.noise_root_dir = '/home/nitya/projects/gpu_projects/amogh/Data/noise'
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sr = sr
        self.clip_samples = clip_samples
        self.threshold = 12
        self.pure_noise = pure_noise
        self.snr_list = snr
        self.flag = flag
        
    def __getitem__(self,index):
        while True:
            notnoise = 1
            # Clean files
            if len(self.clean_files) != 0:
                # Randomly sample a clean file
                f = random.choice(self.clean_files)
                audio, fs = sf.read('{}/{}'.format(self.clean_root_dir,f), dtype='float32')
                # Randomly sample a clean clip
                if(len(audio) > LEN*fs):
                    start = random.randint(0,len(audio)-LEN*fs)
                    clip = resample(audio[start:start+LEN*fs],orig_sr=fs,target_sr=self.sr)
                    
                    mu, sigma = np.mean(clip), np.std(clip)
                    normalized_clean = torch.from_numpy((clip-mu)/(sigma+EPS))
                else:
                    continue
                
            # Noise files
            if len(self.noise_files) != 0:
                nf = random.choice(self.noise_files)
                audio_noise, fs_noise = sf.read('{}/{}'.format(self.noise_root_dir,nf), dtype='float32')
                if len(audio_noise.shape) > 1:
                    audio_noise = np.mean(audio_noise,axis=1)
                audio_noise = audio_noise.astype('float32')
                # Randomly sample a clip of noise
                if len(audio_noise) < LEN*fs_noise: continue
                start = random.randint(0,len(audio_noise)-LEN*fs_noise)
                clip_noise = resample(audio_noise[start:start+LEN*fs_noise],orig_sr=fs_noise,target_sr=self.sr)

                mu_noise, sigma_noise = np.mean(clip_noise), np.std(clip_noise)
                normalized_noise = torch.from_numpy((clip_noise-mu_noise)/(sigma_noise+EPS))
                
                # Mix the noise with the clean audio clip at given SNR level
                r = random.random();
                snr = random.choice(self.snr_list)
                interference = 10**(-snr/20)*normalized_noise

                # ------- original code -------
                # if r < self.pure_noise and self.flag == 'train':
                #     mixture = interference
                # else:
                
                # ------- modified code -------
                mixture = normalized_clean + interference
                
                mu_mixture, sigma_mixture = torch.mean(mixture), torch.std(mixture)
                mixture = (mixture-mu_mixture) / sigma_mixture 

            # --------- original code -----------
            # if len(self.noise_files) != 0:
            #     if self.flag == 'train':
            #         return mixture, normalized_clean, notnoise 
            #     if self.flag == 'test':
            #         return mixture, normalized_clean
            # return normalized_clean

            # --------- modified code -----------

            if len(self.noise_files) != 0:
                if self.flag == 'train':
                    return mixture, normalized_clean, interference, fs_noise, notnoise 
                if self.flag == 'test':
                    return mixture, normalized_clean
            return normalized_clean

    def __len__(self):
        return 1000 # sentinel value

# Get the dataloader for clean, mix, and test
def get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_noise_files=None,test_noise_files=None):
    # Clean
    train_A_data = DapsNoise(train_A_files,[],config['sr'],config['clip_size'],config['pure_noise_a'],config['snr'],'train')
    # Noisy train
    train_B_data = DapsNoise(train_B_files,train_noise_files,config['sr'],config['clip_size'], config['pure_noise_b'],config['snr'],'train')
    # Noisy test
    test_B_data = DapsNoise(test_B_files,test_noise_files,config['sr'],config['clip_size'], config['pure_noise_b'],config['snr'],'test')
    
    train_A_dataloader = DataLoader(train_A_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    train_B_dataloader = DataLoader(train_B_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    test_B_dataloader = DataLoader(test_B_data, batch_size=1, shuffle=True)
    
    test_B_data = []
    for i, audio_pair in enumerate(test_B_dataloader):
        test_B_data.append(audio_pair)
    return train_A_dataloader, train_B_dataloader, test_B_data
