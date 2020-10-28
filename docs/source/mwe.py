import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from evxtb.xtb_ev import xtbev, xtbev_opt, ev_bulk

input_file = "input.vasp"
TO_READ = str(Path(input_file).resolve())

# Set scaling factors that define the range
# of volumes included on the curve
sfs = np.linspace(0.75, 1.05, 15).tolist()

# Get volumes and energies
V, E = xtbev(TO_READ, sfs)

# Plot curve
ev_bulk(V, E, "Example EV Curve")
