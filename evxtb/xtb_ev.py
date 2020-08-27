import re
import subprocess

from ase.eos import EquationOfState
from ase.units import kJ, Hartree, eV
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
    

def run_xtb_opt(fname):
    """Run GFN0 xTB with geometry optimization with to produce "xtbopt.vasp" file and move it to /data.

    Args:
        fname (str): The input atomic coordinate file.
    """
    subprocess.run(["xtb", "--gfn", "0", fname, "-o", "--cycles", "1000"], capture_output=True, text=True)
    #subprocess.run(["mv", "xtbopt.vasp", "./evxtb"], capture_output=True, text=True)


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


def find_repeat_unit(fname):
    """Finds the number of stiochiometric units in the cell.

    Args: 
        fname (str): The vasp coordinate file name.

    Returns:
        rnum (int): The number of repeat units in the cell.

    """
    vasp_handler = VaspHandler(fname)

    rnum = np.gcd.reduce(vasp_handler.read_vasp_repeats())

    return rnum


def xtbev(fname, sfs):
    """Calculate multiple total energies from differently scaled unit cells.

    Args:
        fname (str): The input atomic coordinate file.
        sfs (list of float): The scaling factors that will be applied to the unit cell volume.

    Returns:
        volumes (list of float): The volumes correlated to the energies calculated.
        energies (list of float): The energies the cell calculated by gfn0 xtb then divided by number of repeat chemical units.

    """
    vasp_handler = VaspHandler(fname)
    rnum = find_repeat_unit(fname)
    print("Number of stoichiometric units = ", rnum)

    energies = []
    volumes = []
    for sf in sfs:
        fname = "temp.vasp"
        vasp_handler.write_vasp(sf, fname)

        energies.append(interpret_xtb(run_xtb(fname))/rnum)
        volumes.append(unit_vol(vasp_handler.lat_params) * sf)

    return volumes, energies


def xtbev_opt(fname, sfs):
    """Calculate multiple total energies from differently scaled unit cells.

    Args:
        fname (str): The input atomic coordinate file.
        sfs (list of float): The scaling factors that will be applied to the unit cell volume.

    Returns:
        volumes (list of float): The volumes correlated to the energies calculated.
        energies (list of float): The energies the cell calculated by gfn0 xtb then divided by number of repeat chemical units.

    """
    run_xtb_opt(fname)
    fname = "xtbopt.vasp"

    vasp_handler = VaspHandler(fname)
    rnum = find_repeat_unit(fname)
    print("Number of stoichiometric units = ", rnum)

    energies = []
    volumes = []
    for sf in sfs:
        fname = "temp.vasp"
        vasp_handler.write_vasp(sf, fname)

        energies.append(interpret_xtb(run_xtb(fname))/rnum)
        volumes.append(unit_vol(vasp_handler.lat_params) * sf)

    return volumes, energies


def ev_bulk(volumes, energies, plot_name):
    """Plot volume energy curve and calculates bulk modulus

    Args:
        volumes (list of float): The volumes correlated to the energies calculated.
        energies (list of float): The energies the cell calculated by gfn0 xtb.
        plot_name (str): The file name the plot is saved to. 

    Returns:
        v0 (float): The volume of minimum energy.
        e0 (float): The fitted minimum energy of the V/E curve.
        B (float): The calculated bulk modulus. 

    """
    volumes = np.array(volumes)
    energies = np.array(energies) * Hartree/eV
    eos = EquationOfState(volumes, energies, eos="murnaghan")
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, 'GPa')  # Converts into GPa
    ax = eos.plot()
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(plot_name)
    return v0, e0, B


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

        self.lat_params, self.template = self.read_vasp_lat()

    def read_vasp_lat(self):
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
    
    def read_vasp_repeats(self):
        """Read the class's VASP file and get the numbers of atoms in the cell."""
        with open(self.fname, "r") as f:
            buffer = f.read()
        
        lines = buffer.split("\n")

        elem = lines[6]
        row = [int(num) for num in elem.split()]
        chem_ints = np.array(row)

        return chem_ints


    def write_vasp(self, sf, fname):
        """Write to a VASP file with a scaled lattice parameter.

        Args:
            sf (float): The scaling factor.
            fname (str): The new file to write to.

        """
        scaled = resize_lat(self.lat_params, sf)

        buffer = self.template.format(format_array(scaled))

        with open(fname, "w") as f:
            f.write(buffer)
