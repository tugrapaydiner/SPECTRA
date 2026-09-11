"""Local validation bootstrap: omit the host's NumPy-preloading sitecustomize.

Uses the SAME installed packages; no project guard or arithmetic is changed.
Run with python -S. This file is not part of the SPECTRA implementation.
"""
import sys
sys.path.insert(0, '/opt/pyvenv/lib/python3.13/site-packages')
import runpy
if sys.argv[1] == '-m':
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
else:
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name='__main__')
