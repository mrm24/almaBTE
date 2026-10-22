#!/usr/bin/env python3
"""Optional Phonopy comparison; run with uv and numpy/h5py/phonopy installed.

uv run --with h5py --with phonopy==4.5.0 test/phonopy_reference_test.py BUILD SOURCE
"""
# Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
# Copyright 2026 The ALMA Project Developers
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re,subprocess,tempfile,os,sys
import numpy as np,h5py
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_CONSTANTS
build,root=(Path(arg).resolve() for arg in sys.argv[1:])
program=build/'src/VCAbuilder'
periodic=(root/'src/periodic_table.cpp').read_text()
masses={}
for element,mass,percentage in re.findall(r'\{"([A-Za-z]+)",\s*([\d.]+),\s*([\d.]+)\}',periodic):
    masses[element]=masses.get(element,0.)+float(mass)*float(percentage)/100.
unit=read_vasp(root/'test_resources/InAs/POSCAR')
unit.masses=[masses[e] for e in unit.symbols]
ph=Phonopy(unit,supercell_matrix=[5,5,5],primitive_matrix=np.eye(3),lang='C')
ph.force_constants=parse_FORCE_CONSTANTS(root/'test_resources/InAs/FORCE_CONSTANTS')
born=np.loadtxt(root/'test_resources/InAs/BORN',skiprows=1)
e=1.602176565e-19;epsilon0=8.854187817e-12;amu=1.660538921e-27
# Match almaBTE's physical constants rather than introducing CODATA differences.
params={'born':born[1:].reshape(-1,3,3),'dielectric':born[0].reshape(3,3),'factor':e/(4*np.pi*epsilon0)*1e10}
with tempfile.TemporaryDirectory(prefix='prealma-phonopy-') as temporary:
    tmp=Path(temporary)
    for method in ['wang','gonze']:
        ph.nac_params=dict(params,method=method)
        xml=tmp/'input.xml'
        xml.write_text(f'<singlecrystal><materials_repository root_directory="{root}/test_resources"/><compound name="InAs"/><gridDensity A="3" B="3" C="3"/><target directory="{tmp}"/><nonanalytic_treatment method="{method}"/><overwrite/></singlecrystal>')
        result=subprocess.run([str(program),str(xml)],cwd=tmp,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),text=True,capture_output=True)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        with h5py.File(tmp/'InAs_3_3_3.h5') as h5:
            omega=h5['qpoint_grid/omega'][:]
        max_error=0.
        for i,q in enumerate(np.ndindex(3,3,3)):
            if q==(0,0,0):continue
            ph.dynamical_matrix.run(np.array(q)/3)
            expected=np.linalg.eigvalsh(ph.dynamical_matrix.dynamical_matrix)
            actual=omega[i]*np.abs(omega[i])/(e/amu*1e-4)
            error=np.max(np.abs(actual-expected))/max(1.,np.max(np.abs(expected)))
            max_error=max(max_error,error)
        print(method,'max relative dynamical eigenvalue error',max_error,flush=True)
        if method == 'gonze':
            assert max_error < 2e-7
        # almaBTE intentionally omits the Wang correction on first-BZ faces;
        # Phonopy applies it there. This is a known method convention difference.
        else:
            assert max_error < 2e-4
