import os

files = os.listdir('data/masks')
for dpath, dnames, fnames in os.walk('./data/masks'):
    print(fnames)
    for f in fnames:
        if '_matte' in f:
            os.rename(dpath + '/' + f, dpath + '/' + f.replace('_matte', ''))