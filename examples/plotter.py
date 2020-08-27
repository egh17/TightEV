import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from evxtb.xtb_ev import xtbev, xtbev_opt, ev_bulk


if __name__ == "__main__":
    input_file = "ana_frac_current.vasp"
    TO_READ = str(Path(__file__).parent.parent / "data" / input_file)
    sfs = np.linspace(0.8, 1.2, 15).tolist()
    V, E = xtbev(TO_READ, sfs)
    ev_bulk(V, E, 'plot_2.png')