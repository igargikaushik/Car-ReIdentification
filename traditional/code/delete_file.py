import os

path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/1/'
dirs = os.listdir(path)

for dir in dirs:
    if(os.path.splitext(dir)[1] == '.txt'):
        os.remove(path+dir)