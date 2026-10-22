#!/usr/bin/env python3
"""Optional end-to-end checks: uv run --no-project updates_cli_test.py BUILD SOURCE.

Uses only the Python standard library. BUILD must contain the built applications;
SOURCE must have the standard material fixtures, including _metadata files.
"""
# Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
# Copyright 2026 The ALMA Project Developers
# SPDX-License-Identifier: Apache-2.0
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from xml.sax.saxutils import quoteattr

build, source = (Path(arg).resolve() for arg in sys.argv[1:])
executables = build / "src"
resources = source / "test_resources"
environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

def run(program, *arguments, success=True):
    result = subprocess.run([str(executables / program), *map(str, arguments)],
                            cwd=work, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    if success and result.returncode:
        raise AssertionError(f"{program}: {result.stdout}")
    if not success and result.returncode == 0:
        raise AssertionError(f"{program} unexpectedly succeeded")
    return result.stdout

with tempfile.TemporaryDirectory(prefix="prealma-updates-cli-") as temporary:
    work = Path(temporary)
    output = work / "materials" / "Si"
    xml = work / "material.xml"
    xml.write_text(f'''<singlecrystal>
      <materials_repository root_directory={quoteattr(str(resources))}/>
      <compound name="Si"/><gridDensity A="3" B="3" C="3"/>
      <target directory={quoteattr(str(output))}/><overwrite/>
    </singlecrystal>''')
    run("VCAbuilder", xml)
    native = output / "Si_3_3_3.h5"
    assert native.is_file()
    run("gruneisen", xml, "300")
    gruneisen = list(output.glob("*.gruneisen"))
    assert len(gruneisen) == 1
    rows = list(csv.reader(line for line in gruneisen[0].read_text().splitlines()
                           if not line.startswith("#")))
    assert len(rows) == 27 * 6
    assert all(math.isfinite(float(value)) for row in rows for value in row)
    sweep = work / "sweep.xml"
    sweep.write_text(f'''<Tsweep>
      <H5repository root_directory={quoteattr(str(work / 'materials'))}/>
      <compound directory="Si" base="Si" gridA="3" gridB="3" gridC="3"/>
      <sweep type="linear" start="300" stop="400" points="2"/>
      <computeCoherenceTerm/>
      <target directory={quoteattr(str(work / 'results'))} file="conductivity.csv"/>
    </Tsweep>''')
    run("kappa_Tsweep", sweep)
    coherence = work / "results" / "conductivity.csv_coherence"
    with coherence.open() as stream:
        result = list(csv.reader(stream))
    assert len(result) == 3 and len(result[0]) == 7
    assert all(math.isfinite(float(value)) for row in result[1:] for value in row)
    assert max(float(result[1][column]) for column in (1, 4, 6)) >= 0.0
    interface = work / "interface.xml"
    values = []
    for probability in (.25, .5):
        interface.write_text(f'''<interface_conductance>
          <materialA><material directory={quoteattr(str(output))} database="Si_3_3_3.h5"/>
            <axis x="1" y="0" z="0"/></materialA>
          <materialB><material directory={quoteattr(str(output))} database="Si_3_3_3.h5"/>
            <axis x="1" y="0" z="0"/></materialB>
          <Temperature T="300"/><Interface model="constant" alpha="{probability}"/>
        </interface_conductance>''')
        text = run("interfaces", interface)
        values.append(float(re.search(r"G : (\S+)", text)[1]))
    assert values[0] > 0 and math.isclose(2 * values[0], values[1], rel_tol=1e-5)
    # A converted original fixture still lacks the operator required for coherence.
    original = resources / "Si" / "Si_12_12_12.h5"
    if original.is_file():
        converted = work / "legacy" / "Si"
        converted.mkdir(parents=True)
        run("convert_bulk_hdf5", original, converted / original.name)
        sweep.write_text(sweep.read_text().replace(str(work / "materials"), str(work / "legacy"))
                         .replace('gridA="3" gridB="3" gridC="3"', 'gridA="12" gridB="12" gridC="12"'))
        text = run("kappa_Tsweep", sweep, success=False)
        assert "coherence requires Wigner velocities" in text
    print("CLI checks passed: material builder, Gruneisen, coherence output, interface scaling, legacy availability.")
