import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qiskit.circuit import (QuantumCircuit,
                            QuantumRegister, ClassicalRegister,
                            Parameter, ParameterVector)
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Choi
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, SPSA, COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import state_fidelity,DensityMatrix
from qiskit import Aer, execute

def swap_test(qc, ref_state, state1, state2):
    qc.h(ref_state)
    qc.cswap(control, state1, state2)
    qc.h(ref_state)
    return qc


def cost_func_domain(params_values):
    probabilities = qnn.forward([], params_values)
    # we pick a probability of getting 1 as the output of the network
    cost = np.sum(probabilities[:, 1])

def matrix_square_root(matrix):
    evalues, evectors = np.linalg.eig(a)
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix

#def fidelity(state1, state2t):
#    fidelity = np.trace()**2
#    return fidelity


num_qbits = 4
num_cbits = 0
num_trash = 2
qr = QuantumRegister(num_qbits)
cr = ClassicalRegister(num_cbits)
qc = QuantumCircuit(qr, cr)

train_circuit = QuantumCircuit(4)
train_circuit.x(0)

encoder_u = RealAmplitudes(4, entanglement='full', reps=1, name='u1')
encoder_v = RealAmplitudes(4, entanglement='full', reps=1, name='v1')
#pars_u = ParameterVector(name='u_theta', length=encoder_u.num_parameters)
pars_u = [0.0] * encoder_u.num_parameters
#pars_v = ParameterVector(name='v_theta', length=encoder_v.num_parameters)
pars_v = [0.0] * encoder_v.num_parameters
encoder_u.assign_parameters(pars_u, inplace=True)
encoder_v.assign_parameters(pars_v, inplace=True)

init_theta = np.random.rand(encoder_u.num_parameters)

qc.compose(encoder_u, qr, inplace=True)
qc.barrier()
qc.compose(train_circuit, qr, inplace=True)
qc.barrier()
qc.compose(encoder_v, qr, inplace=True)

#qc.draw('mpl')
#plt.show()
choi_state = Choi(qc)
epilson_choi = Choi(train_circuit)
print(type(choi_state))
print(state_fidelity(DensityMatrix.from_instruction(choi_state.to_instruction()),
                     DensityMatrix.from_instruction(choi_state.to_instruction())))
backend = Aer.get_backend('statevector_simulator')

#sv1 = execute(qc, backend).result().get_statevector(qc)
#print(sv1)
#state_fidelity()

# l3 = 1-1/N*()**2

#qnn = SamplerQNN(circuit=qc,
#                 input_params=[],
#                 weight_params=ae.parameters,
#                 interpret=identity_interpret,
#                 output_shape=2,
#                 )
#
#optimizer = COBYLA()
#
#initial_point = algorithm_globals.random.random(ae.num_parameters)
#opt_result = optimizer.minimize(cost_func_domain, initial_point)





