#!/usr/bin/env python
import os
import yaml
import numpy as np
import h5py

ms = list(range(100, 112)) + list(range(113, 126))

os.chdir(os.path.expandvars('$xveganx/sf/Anon1/'))

#for ch in chs:
#    os.chdir("K_ch{:03d}".format(ch))
#    os.system('mkdir libraries &')
#    os.system('grid.py --create > grid.out &')
#    os.chdir("..")

for m in ms:
    os.chdir("m{:03d}".format(m))
    os.system('$Starfish/scripts/pca.py --create > pca_create.out')
    os.system('time $Starfish/scripts/pca.py --optimize=emcee --samples=40  > pca_optimize.out')
    os.system('time $Starfish/scripts/pca.py --store --params=emcee > pca_store.out')
    os.system('mkdir output')
    os.system('mkdir output/mix_emcee')
    os.system('mkdir output/mix_emcee/run01')
    os.system('cp $xveganx/sf/Anon1/m112/output/mix_emcee/run01/s0_o0phi.json output/mix_emcee/run01/')
    os.system('cp $xveganx/sf/Anon1/m112/user_prior.py .')
    os.system('cp config.yaml output/mix_emcee/run01/')
    os.chdir("..")