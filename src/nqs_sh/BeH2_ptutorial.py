import netket as nk
import netket.experimental as nkx
import pickle
import numpy as np
import math
from functools import reduce



# Unitary matrices for the rotation in the X and Y bases
rotationX = 1./(math.sqrt(2))*np.asarray([[1.,1.],[1.,-1.]])
rotationY= 1./(math.sqrt(2))*np.asarray([[1.,-1j],[1.,1j]])

# Explicitly define the pauli matrices
# Identity 
I = np.asarray([[1.,0.],[0.,1.]])
# Pauli X
X = np.asarray([[0.,1.],[1.,0.]])
# Pauli Y
Y = np.asarray([[0.,-1j],[1j,0.]])
# Pauli Z
Z = np.asarray([[1.,0.],[0.,-1.]])

# Functions to load data measurement data - sigmas + Us - this one doesn't load Us properly so use the 2nd load bases
def LoadData(N,hilbert,path_to_samples, path_to_bases):
    training_samples = []
    training_bases = []

    tsamples = np.loadtxt(path_to_samples)
    assert(N == tsamples.shape[1])
    fin_bases = open(path_to_bases, 'r')
    lines = fin_bases.readlines()

    bases = []

    for b in lines:
        basis = ""
        assert(len(b) == N + 1)
        for j in range(N):
            basis += b[j]
        bases.append(basis)
    index_list = sorted(range(len(bases)), key=lambda k: bases[k])
    bases.sort()

    for i in range(len(tsamples)):
        training_samples.append(tsamples[index_list[i]].tolist())

    rotations = []

    tmp = ''
    b_index = -1
    for b in bases:
        if (b != tmp):
            tmp = b
            localop = nk.operator.LocalOperator(hilbert, 1.0)

            for j in range(N):
                if (tmp[j] == 'X'):
                    localop = localop * nk.operator.LocalOperator(hilbert, rotationX, [j])
                if (tmp[j] == 'Y'):
                    localop = localop * nk.operator.LocalOperator(hilbert, rotationY, [j])

            rotations.append(localop)
            b_index += 1
        training_bases.append(b_index)

    return tuple(rotations), np.asarray(training_samples), np.asarray(training_bases)

def OperatorFromString(op_string):                                                
    OpList = []                                                                   
    Sites = []                                                                    
    for k in range(len(op_string)):                                               
        if (op_string[k] == 'X'):                                                 
            OpList.append(X)                                                      
            Sites.append(k)                                                       
        elif (op_string[k] == 'Y'):                                               
            OpList.append(Y)                                                      
            Sites.append(k)                                                       
        elif (op_string[k] == 'Z'):                                               
            OpList.append(Z)                                                      
            Sites.append(k)
        elif (op_string[k] == 'I'):  # treat I same as others so sites is built properly                                    
            OpList.append(I)                                                      
            Sites.append(k)                                                     
    return Sites,reduce(np.kron,OpList) 

# Build the hamiltonian using the pauli words and coeffecients
def BuildHamiltonian(N,hilbert,pauli_path,interactions_path):                     
    pauli = np.load(pauli_path,allow_pickle=True)                                 
    interactions = np.load(interactions_path,allow_pickle=True)                   
                                                                                  
    hamiltonian = nk.operator.LocalOperator(hilbert, 0.0)                                                                              
    for h in range(0,len(pauli)):                                                                                           
        # treat Identity ops the same as other ops                                                                   
        sites,operator = OperatorFromString(pauli[h])                         
        h_term = interactions[h]*operator
        hamiltonian = hamiltonian + nk.operator.LocalOperator(hilbert,h_term,sites)       
            
    return hamiltonian 


# bases had an error: so made new function
def load_bases(filename: str) -> list:
    """
    Load measurement bases from a text file.
    
    Args:
        filename: Path to bases file (e.g., "bases.txt")
                  Each line should contain a basis string like "XZZY" or "ZZIZ"
    
    Returns:
        Us: List of measurement bases as strings
            e.g., ['XZZY', 'ZZIZ', 'XYZI', ...]
    
    Example:
        >>> Us = load_bases("bases.txt")
        >>> print(Us[0])  # 'XZZY'
        >>> print(len(Us[0]))  # 4 (number of qubits)
    """
    Us = []
    
    with open(filename, 'r') as f:
        for line in f:
            basis_str = line.strip()  # Remove whitespace/newlines
            
            # Validate that it only contains valid Pauli operators
            if not all(c in 'XYZI' for c in basis_str):
                print(f"Warning: Invalid basis string '{basis_str}' - skipping")
                continue
            
            Us.append(basis_str)
    
    print(f"Loaded {len(Us)} measurement bases from {filename}")
    if len(Us) > 0:
        print(f"Number of qubits: {len(Us[0])}")
        
        # Show statistics
        from collections import Counter
        basis_counts = Counter(Us)
        print(f"Unique bases: {len(basis_counts)}")
        if len(basis_counts) <= 10:
            print(f"Basis distribution:")
            for basis, count in sorted(basis_counts.items()):
                print(f"  {basis}: {count}")
    
    return Us

def MeasurementRotationFromString(basis_string):
    """
    Convert basis string to rotation operator matrix for measurements.
    
    For measurement bases, we need rotation unitaries, not Pauli operators:
    - I or Z: Identity (no rotation, measure in computational basis)
    - X: Hadamard (rotation to X eigenbasis)
    - Y: S†H (rotation to Y eigenbasis)
    """
    
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
    U_Y = H_gate @ S_dag
    
    rotation_map = {
        'I': I,
        'Z': I,      # No rotation for Z basis
        'X': H_gate, # Hadamard for X basis
        'Y': U_Y     # S†H for Y basis (flip order)
    }
    
    OpList = []
    Sites = []
    
    for k, char in enumerate(basis_string):
        if char in rotation_map:
            OpList.append(rotation_map[char])
            Sites.append(k)
        else:
            raise ValueError(f"Unknown basis character: {char}")
    
    return Sites, reduce(np.kron, OpList)


# Function to build the base operators properly -- SAVIOUR RIGHT HERE!
def BuildBases(hilbert,bases_array):
    base_ops = []
    
    for basis_string in bases_array:
        basis_string = str(basis_string)
        sites,operator = MeasurementRotationFromString(basis_string)
        base_operator = nk.operator.LocalOperator(hilbert,operator,sites)
        base_ops.append(base_operator)
    return np.array(base_ops,dtype=object)


# Correction to earlier version: use * directly
# def BuildBases(hilbert, bases_array):
    
#     """Build basis operators using * """
#     H_gate = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
#     S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
#     U_Y = H_gate @ S_dag
#     I_matrix = np.eye(2, dtype=complex)
    
#     base_ops = []
    
#     print("used new bb function")
    
#     for basis_string in bases_array:
#         basis_string = str(basis_string)
        
#         localop = None
#         for j in range(len(basis_string)):
            
#             if basis_string[j] == 'X':
#                 op = nk.operator.LocalOperator(hilbert, H_gate, [j])
#             elif basis_string[j] == 'Y':
#                 op = nk.operator.LocalOperator(hilbert, U_Y, [j])
#             elif basis_string[j] in ['Z', 'I']:
#                 op = nk.operator.LocalOperator(hilbert, I_matrix, [j])  # Identity for both
#             else:
#                 continue
            
#             if localop is None:
#                 localop = op
#             else:
#                 localop = localop * op
        
#         base_ops.append(localop)
    
#     return np.array(base_ops, dtype=object)