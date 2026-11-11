from pathlib import Path
import ctypes
import numpy as np
LIB=ctypes.CDLL(str(Path(__file__).resolve().parent/'gates.dll'))
DOUBLE=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
INT=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
LIB.evolve.argtypes=[DOUBLE,ctypes.c_int,ctypes.c_int,ctypes.c_int,INT,INT,INT,DOUBLE,DOUBLE]
LIB.evolve.restype=ctypes.c_int
def evolve_ansatz(states,operations):
 if states.dtype!=np.complex128 or not states.flags.c_contiguous:raise ValueError('Native gates require C-contiguous complex128')
 kind=np.array([int(name!='ry') for name,q,v in operations],dtype=np.int32)
 qubit=np.array([q for name,q,v in operations],dtype=np.int32)
 target=np.array([int(v) if name!='ry' else 0 for name,q,v in operations],dtype=np.int32)
 cosine=np.array([np.cos(v/2) if name=='ry' else 0 for name,q,v in operations],dtype=np.float64)
 sine=np.array([np.sin(v/2) if name=='ry' else 0 for name,q,v in operations],dtype=np.float64)
 code=LIB.evolve(states.view(np.float64),len(states),states.shape[1],len(operations),kind,qubit,target,cosine,sine)
 if code:raise RuntimeError('Native evolution failed: '+str(code))
 return states
