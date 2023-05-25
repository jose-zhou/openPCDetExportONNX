import os, glob
from unittest import result
filePath = '/media/jose/6842930A30C05552/3Drecgonization/code/OpenPCDet/data/mykitti/training/label'

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
        return result

files_full_path = all_path(filePath)
classname = []
for txt in files_full_path:
    with open(txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        label = line.strip().split(' ')
        tempName = label[0]
        if not (classname.__contains__(tempName)):
            classname.append(tempName)

print(classname)
   

