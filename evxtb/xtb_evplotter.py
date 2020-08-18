import re
import subprocess

from tempfile import NamedTemporaryFile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def interpret_xtb(xtb_output):
    """Find the total energy from the xtb output text.

    Args:
        xtb_output (str): The output from xtb.

    Reurns:
        energy (float): The total energy value given by xtb.

    """
    pattern = r"TOTAL ENERGY\s+(-?\d+\.\d+)"
    match = re.search(pattern, xtb_output)

    return float(match.group(1))


def read_vasp(fname):
    with open(fname, "r") as f:
        buffer = f.read()

    lines = buffer.split("\n")
    elems = lines[2:5]
    rows = [[float(num) for num in line.split()] for line in elems]
    return np.array(rows)


def resize_lat(array, scale):
    """Scale a lattice's volume by a given factor.

    Args:
        array (:obj:`np.ndarray`): The lattice parameters.
        scale (float): The scaling factor.

    Returns:
        scaled (:obj:`np.ndarray`): The scaled lattice.

    """
    return scale ** (1 / 3) * array


def format_array(array):
    """Format an array nicely for templating.

    Removes the square brackets and commas.

    Args:
        array (:obj:`np.ndarray`): The array to format.

    Returns:
        formatted (str): The formatted array.

    """
    formatted = ""
    template = "{}\t{}\t{}\n"
    for rows in array:
        formatted += template.format(*rows)

    formatted = formatted.rstrip("\n")
    return formatted


def run_xtb(fname):
    """Run GFN0 xTB with given file.
    
    Args:
        fname (str): The input atomic coordinate file.

    Returns:
        xtb results (str): The xtb code output. 

    """
    res = subprocess.run(["xtb", "--gfn", "0", fname], capture_output=True, text=True)
    return res.stdout


def unit_vol(lat_params):
    """Calculate volume of unit cell from lattice parameter matrix.

    Args:
        lat_params (:obj:`np.ndarray`): The lattice matrix.

    Returns:
        volume (float): The cell volume.

    """
    a = lat_params[0, :]
    b = lat_params[1, :]
    c = lat_params[2, :]
    return np.dot(np.cross(a, b), c)


def xtbev(fname, sfs):
    """Calculate multiple total energies from differently scaled unit cells.

    Args:
        fname (str): The input atomic coordinate file.
        sfs (list of float): The scaling factors that will be applied to the unit cell volume.

    Reurns:
        volumes (list of float): The volumes correlated to the energies calculated
        energies (list of float): The energies the cell calculated by gfn0 xtb

    """
    vasp_handler = VaspHandler(fname)

    energies = []
    volumes = []
    for sf in sfs:
        fname = "temp.vasp"
        vasp_handler.write_vasp(sf, fname)

        energies.append(interpret_xtb(run_xtb(fname)))
        volumes.append(unit_vol(vasp_handler.lat_params) * sf)

    return volumes, energies


class VaspHandler:
    """Handler for VASP inputs and lattice scaling.

    Args:
        fname (str): The input template file.

    Attributes:
        fname (str): The input template file.
        lat_params (:obj:`np.ndarray`): The lattice matrix.
        template (str): The template of the VASP file, sans lattice parameter.

    """

    def __init__(self, fname):
        self.fname = fname

        self.lat_params, self.template = self.read_vasp()

    def read_vasp(self):
        """Read the class's VASP file and get the lattice parameter."""
        with open(self.fname, "r") as f:
            buffer = f.read()

        lines = buffer.split("\n")

        # Get the lattice parameters
        elems = lines[2:5]
        rows = [[float(num) for num in line.split()] for line in elems]
        lat_params = np.array(rows)

        # Get the rest of the template
        rest = lines[:2] + ["{}"] + lines[5:]
        template = "\n".join(rest)

        return lat_params, template

    def write_vasp(self, sf, fname):
        """Write to a VASP file with a scaled lattice parameter.

        Args:
            sf (float): The scaling factor.
            fname (str): The file to write to.

        """
        scaled = resize_lat(self.lat_params, sf)

        buffer = self.template.format(format_array(scaled))

        with open(fname, "w") as f:
            f.write(buffer)


if __name__ == "__main__":
    TO_READ = str(Path(__file__).parent.parent / "data" / "Rutile.vasp")
    sfs = np.linspace(0.5, 1.5, 15).tolist()
    V, E = xtbev(TO_READ, sfs)
    plt.plot(V, E)
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Energy (Eh)")
    plt.title("Rutile EV")
    plt.savefig("Rutile_EV.png")