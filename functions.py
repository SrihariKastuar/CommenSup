import numpy as np
from ase import Atoms
from ase.io import write

def supercell_vectors(M, N, a1, a2):
    T1 = M * a1 + N * a2
    T2 = -N * a1 + (M + N) * a2
    return T1, T2

def rotmat(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    return np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                     [np.sin(theta_rad),  np.cos(theta_rad)]])
    
def supercell_vectors_layer2per(Mp, Np, a1, a2, theta_deg, delta):
    R = rotmat(theta_deg)
    a1p = (1 + delta) * R @ a1
    a2p = (1 + delta) * R @ a2
    T1p = Mp * a1p + Np * a2p
    T2p = -Np * a1p + (Mp + Np) * a2p
    return T1p, T2p

def graphene_lattice_vectors(a):
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2])
    return a1, a2

def rotate_and_scale(v, theta_deg, delta):
    R = rotmat(theta_deg)
    return (1 + delta) * R @ v

def offset_A_sublattice():
    return np.array([0.0, 0.0])

def offset_B_sublattice(a1, a2):
    return (a1 + a2) / 3.0

def supercell_search(theta_input_deg, delta_input, tol_angle_deg, tol_delta, Umax=12):
    matches = []
    theta_input_deg = float(theta_input_deg)
    delta_input = float(delta_input)
    rng = range(-Umax, Umax+1)
    for M in rng:
        for N in rng:
            if M == 0 and N == 0:
                continue
            S1 = M**2 + M*N + N**2
            for Mp in rng:
                for Np in rng:
                    if Mp == 0 and Np == 0:
                        continue
                    S2 = Mp**2 + Mp*Np + Np**2
                    if S2 == 0:
                        continue
                    l_ratio = np.sqrt(S1 / S2)
                    delta_eff = l_ratio - 1.0
                    ctheta = (2*M*Mp + N*Mp + M*Np + 2*N*Np) / (2 * np.sqrt(S1) * np.sqrt(S2))
                    ctheta = np.clip(ctheta, -1.0, 1.0)
                    theta_eff_rad = np.arccos(ctheta)
                    theta_eff_deg = np.degrees(theta_eff_rad)
                    # Compare to input in degrees!
                    if abs(delta_eff - delta_input) <= tol_delta and abs(theta_eff_deg - theta_input_deg) <= tol_angle_deg:
                        area = (S1+S2)/2  #abs(M*Np - N*Mp)
                        if area > 0:
                            matches.append({
                                "M": M, "N": N, "Mp": Mp, "Np": Np,
                                "theta_eff_deg": theta_eff_deg,
                                "delta_eff": delta_eff,
                                "area": area, 
                                "atom_count": 4 * area # 2 atoms per layer x2 layers
                        })
    return matches


    
def angle_deg(v1, v2):
    return np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def points_in_moire_cell(a1, a2, tau, T1, T2):
    T = np.column_stack((T1[:2], T2[:2]))
    A = np.column_stack((a1, a2))

    Tinv = np.linalg.inv(T)
    Ainv = np.linalg.inv(A)

    C = Ainv @ T
    print("graphene to moire supercell transformation matrix:", C)
    C_int = np.rint(C).astype(int)
    #print(C_int)
    atol = 1e-5
    if not np.allclose(C, C_int, atol=atol):
        raise ValueError(f"Maybe not a commensurate cell (C tolerance is {atol})")
    #N = np.linalg.det(C_int)
    N = int(round(np.linalg.det(C_int)))
    print(f"No. of sites for offset {tau} = {N}")

    corners = np.array([[0,0], C_int[:,0], C_int[:,1], C_int[:,0]+C_int[:,1]], dtype=int)
    #print(corners)
    i_min, j_min = corners.min(axis=0)
    i_max, j_max = corners.max(axis=0)
    print(i_min, j_min, i_max, j_max)
    
    Cinv = np.linalg.inv(C_int.astype(float))

    cart_list, frac_list = [], []

    for i in range(i_min, i_max+1): 
        for j in range(j_min, j_max+1):
            x = Cinv @ np.array([i, j], dtype=float)
            if 0 <= x[0] < 1 and 0 <= x[1] <1:
                r = A @ np.array([i,j], float) + tau
                f = Tinv @ r 
                f = f - np.floor(f)
                cart_list.append(r)
                frac_list.append(f)

    if not cart_list:
        return np.empty((0,2)), np.empty((0,2))

    cart = np.array(cart_list)
    frac = np.array(frac_list)

    key = np.round(frac, 12)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)[:N]
    return cart[uniq_idx], frac[uniq_idx]


def write_poscar(filename, T1, T2, fracA, fracB, fracA2, fracB2, species=("C", "C", "B", "N"), c=25.0, h=3.35):
    cell = [[*T1, 0], [*T2, 0], [0, 0, c]]
    xy = np.vstack([fracA, fracB, fracA2, fracB2])
    z1 = 0.5 - h/(2*c)
    z2 = 0.5 + h/(2*c)
    z = ([z1] * (len(fracA) + len(fracB)) + [z2] * (len(fracA2) + len(fracB2)))
    symbols = []
    for s, coords in zip(species, [fracA, fracB, fracA2, fracB2]):
        symbols.extend([s] * len(coords))
    atoms = Atoms(symbols=symbols, scaled_positions=np.column_stack((xy, z)), cell=cell, pbc=True)
    write(filename, atoms, format="vasp", direct=True, sort=True)
    return atoms

