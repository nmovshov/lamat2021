# Set workspace (paths and/or common variables).

import sys, os

if os.name == 'nt':
    hd = os.getenv('userprofile')
else:
    hd = os.getenv('HOME')

rootd = hd+os.sep+'GitHub'+os.sep+'lamat2021'+os.sep
sys.path.append(rootd)
sys.path.append(rootd+os.sep+'..'+os.sep)
del hd

import observables
import lamat2021 as l21