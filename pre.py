import pandas as pd
import numpy as np
import os, librosa, soundfile

def ffmpegfunc(in_file, out_file):
    y, sr = librosa.load(in_file, sr=None)
    y_16k = librosa.resample(y, sr, 16000)
    soundfile.write(out_file, y_16k, 16000)
    
def emolarge():
    path = '/home/xuxiao/workbench/DeepL/ffmpegaudio/'
    tar = '/home/xuxiao/workbench/DeepL/opensmileaudio/'
    config = r'/home/xuxiao/workbench/DeepL/opensmile-3.0-linux-x64/config/misc/emo_large.conf'
    opensmilepath = r'/home/xuxiao/workbench/DeepL/opensmile-3.0-linux-x64/bin/SMILExtract'
    for item in os.listdir(path):
        os.system(opensmilepath + ' -C ' + config + ' -I ' + path+item + ' -O ' + tar + item[:-4] + '.txt')

def generateEmoLarge():
    res = []
    opensmileResPath = r'/home/xuxiao/workbench/DeepL/opensmileaudio/'
    for item in os.listdir(opensmileResPath):
        f = open(opensmileResPath + item,'r')
        ls = []
        for line in f:
            ls.append(line.strip('\n'))
        itemFeature = ls[6559].split(',')
        itemFeature[0] = item[:-6] if len(item)>=10 else item[:-4]
        res.append(itemFeature)
        f.close()
    csv001 = open(opensmileResPath+'1.txt' , 'r')
    head = []
    for line in csv001:
        head.append(line.strip('\n').replace('@attribute','').replace('numeric','').replace('string','').replace(' ', ''))
    head = head[2:6556]
    my_df = pd.DataFrame(res)
    my_df.to_csv('emo_large_res.csv', index=False, header=head)

def diffcsv():
    df = pd.read_csv('emo_large_res.csv').reset_index(drop=True)
    for i in range(len(df)):
        tdf = df.iloc[i]
        tdf.to_csv('image-features/'+tdf['name']+'/emolarge.csv', index=False, header=True)

def addlabel():
    hdf1 = pd.read_excel('HAMD（13-24）-155.xlsx',sheet_name='病人组',engine='openpyxl')
    hdf2 = pd.read_excel('HAMD（13-24）-155.xlsx',sheet_name='健康组',engine='openpyxl')
    hdf = pd.concat([hdf1,hdf2],axis=0).reset_index(drop=True)[['group','standard_id']]
    print(hdf)
    pdf1 = pd.read_excel('PHQ-9（13-24岁）-108.xlsx',sheet_name='病人组',engine='openpyxl')
    pdf2 = pd.read_excel('PHQ-9（13-24岁）-108.xlsx',sheet_name='健康组',engine='openpyxl')
    pdf = pd.concat([pdf1,pdf2],axis=0).reset_index(drop=True)[['group','standard_id']]
    print(pdf)
    df = pd.concat([hdf,pdf],axis=1).reset_index(drop=True)
    print(df)

# path = '/home/xuxiao/workbench/DeepL/杨文举论文健康人补充语音备份/'
# tar = '/home/xuxiao/workbench/DeepL/ffmpegaudio/'
# for item in os.listdir(path):
#     ffmpegfunc(path+item,tar+item)
# emolarge()
# generateEmoLarge()
# diffcsv()
addlabel()