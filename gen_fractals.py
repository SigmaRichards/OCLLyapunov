import subprocess
import sys
from tqdm import tqdm

all_perms = []
maxv = 2**8
for a in range(2,maxv):
    if (a+1) & a == 0: #If power of 2
        continue;
    cbin = bin(a)[2:]
    all_perms.append(cbin)
all_perms = [a.replace('',',')[1:-1] for a in all_perms]

process = "./DO-THE-ROAR"
path = "./samples/"

for a in tqdm(all_perms):
    default_args = [*sys.argv,
                    "-s",a,
                    "-o",path+a.replace(",","")+".png"]
    subprocess.call([process,*default_args])
