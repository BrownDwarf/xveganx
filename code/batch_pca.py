#!/usr/bin/env python
import os
import yaml
import numpy as np
import h5py

ms = list(range(106, 125))
#ms = list(range(105, 106))

os.chdir(os.path.expandvars('$xveganx/sf/V827Tau/'))

for m in ms:
    os.chdir("m{:03d}".format(m))
    os.system('$Starfish/scripts/pca.py --create > pca_create.out')
    os.system('time $Starfish/scripts/pca.py --optimize=emcee --samples=3  > pca_optimize.out')
    os.system('time $Starfish/scripts/pca.py --store --params=emcee > pca_store.out')
    os.system('mkdir output')
    os.system('mkdir output/mix_emcee')
    os.system('mkdir output/mix_emcee/run01')
    os.system('cp $xveganx/sf/V827Tau/m104/output/mix_emcee/run01/s0_o0phi.json output/mix_emcee/run01/')
    #os.system('cp $xveganx/sf/V827Tau/m104/user_prior.py .')
    os.system('cp config.yaml output/mix_emcee/run01/')
    os.chdir("..")