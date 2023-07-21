from ase import Atom, Atoms
from ase.io import read, iread, write
import numpy as np
from scipy.optimize import minimize

def xyz_to_clmb(atoms, max_atoms=50):
    """
    This updated function takes Cartesian coordinates object as input and converts it into a Coulomb Matrix.
    Coulomb Matrix is formulated as follow:
        \begin{equation}
        M_{ij}=\left\{
        \begin{matrix}
        0.5 Z_i^{2.4} & \text{for } i = j \\
            \frac{Z_i Z_j}{R_{ij}} & \text{for } i \neq j
        \end{matrix}
        \right.
    \end{equation}

    The diagonal elements are a polynomial fit of the atomic energies to the nuclear charge Z. Off-diagonal elements are repultion between nuclei i and j. 

    Args:
        cords (numpy.ndarray): An array of 3D coordinates with shape (N, 3).
        max_atoms: An integer of which indicate the size of the Coulomb Matrix. 

    Returns:
        numpy.ndarray: A 2D array containing the elemnts of Coulomb Matrix as formulated above.

    """
    n_atoms = len(atoms)
    if n_atoms > max_atoms:
        raise ValueError("Number of atoms exceeds max_atoms limit.")
    coulomb_matrix = np.zeros((max_atoms, max_atoms))
    
    def get_atoms():
        return atoms.get_chemical_symbols()
    
    def get_coord():
        return atoms.get_positions()
    
    def get_atomic_number():
        return atoms.get_atomic_numbers()
  
    def calculate_distance(atom1_index, atom2_index):
        atom1_coords = atoms[atom1_index].position
        atom2_coords = atoms[atom2_index].position
        distance = np.linalg.norm(atom1_coords - atom2_coords)
        return distance
    
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            number = get_atomic_number()
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atoms[i].number ** 2.4
            else:
                distance = calculate_distance(i, j)
                coulomb_matrix[i, j] = atoms[i].number * atoms[j].number / distance
                coulomb_matrix[j, i] = coulomb_matrix[i, j]
    
    return coulomb_matrix




def get_D_init(cords):
    """
    Calculate the pairwise distances between points in the initial coordinates.

    Args:
        cords (numpy.ndarray): An array of 3D coordinates with shape (N, 3).

    Returns:
        numpy.ndarray: A 2D array containing the pairwise distances between the input points.
    """
    def distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    D = []
    for i in cords:
        D.append([])
        for j in cords:
            D[-1].append(distance(i, j))

    return np.array(D)

def minimize_D(D_ts, coords, atoms=None):
    """
    Optimization of the transition state synthetic coordinates by utilizing the default parameters 
    from SciPy optimize.minimize that uses the BFGS method for optimization. 
    The number of steps depends on its default setting of convergence criteria of 1×10^{−5}.
    The TS guess coordinates for each atom are determined by reshaping the (1×3N)C_ts vector
    to (N×3)C_TS vector, where each column is the coordinate for each atom. 

    Args:
        D_ts (numpy.ndarray): A 2D array of target distances.
        coords (numpy.ndarray): An array of 3D coordinates with shape (N, 3).
        atoms (list or None, optional): List of atom labels for each coordinate.
                                       If provided, a temporary file 'temp_mov.xyz' will be created.

    Returns:
        float: The loss value.
    """
    print("^^Entering D matrix^^")

    if atoms:
        g = open('temp_mov.xyz', 'w')
        
    def loss(x0):
        """
        The loss function 
        l = \sum_{i=1}^{N}\sum_{i=j}^{N}(D_{ij}^{init} - D_{ij}^{ts})^2

        Returns the trajectory of the TS coordinates.
        """
        crd = x0.reshape((-1, 3))
        if atoms:
            g.write(str(len(crd)) + '\nMinimization of TS-GAN generated structures.\n')
            for j in range (len(atoms)):
                g.write(atoms[j] + ' ' + ' '.join(list(map(str, crd[j]))) + '\n')
        D_init = get_D_init(crd)

        loss_f = sum((D_init.flatten() - D_ts.flatten())**2)

        return loss_f


    x0 = np.copy(coords).flatten()

    loss_f = minimize(loss, x0).x

    g.close()

    loss_f = loss_f.reshape((-1, 3))

    return loss_f






