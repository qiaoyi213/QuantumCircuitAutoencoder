import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import (QuantumCircuit,
                            QuantumRegister, ClassicalRegister,
                            Parameter, ParameterVector)
from qiskit.quantum_info import state_fidelity, DensityMatrix, partial_trace
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, SPSA, COBYLA
from qiskit import Aer, execute


def encoder(num_bits, reps, name):
    encoder = RealAmplitudes(num_bits, entanglement='full', reps=reps, name=name)
    return encoder


def encode_qc(num_qbits, num_trash, train_circuit, encoder_u, encoder_v, pars_u, pars_v):
    qr = QuantumRegister(num_qbits)
    qc_encode = QuantumCircuit(qr)
    u1 = encoder_u.assign_parameters(pars_u)
    v1 = encoder_v.assign_parameters(pars_v)
    qc_encode.compose(u1, qr, inplace=True)
    qc_encode.compose(train_circuit, qr, inplace=True)
    qc_encode.compose(v1, qr, inplace=True)
    return qc_encode


def swap_test(qc, cbit, ref_states, trash_states, aux_state):
    qc.h(aux_state)
    for i_ref in ref_states:
        for i_trash in trash_states:
            qc.cswap(aux_state, i_ref, i_trash)
    qc.h(aux_state)
    qc.measure(aux_state, cbit)
    return qc


def fidelity(qc, train_circuit):
    choi_state = Choi(qc)
    epilson_choi = Choi(train_circuit)
    fidelity = state_fidelity(DensityMatrix.from_instruction(choi_state.to_instruction()),
                              DensityMatrix.from_instruction(epilson_choi.to_instruction()))
    return fidelity


def loss_fun(theta):

    return loss


train_circuit = QuantumCircuit(4)
train_circuit.x(1)
D_train = [train_circuit]


num_qbits = 4
num_cbits = 1
num_trash = 1
num_aux = 1
qr = QuantumRegister(num_qbits - num_trash)
trash_r = QuantumRegister(num_trash, 'trash')
ref_r = QuantumRegister(num_trash, 'ref')
aux_r = QuantumRegister(num_aux, 'aux')
cr = ClassicalRegister(num_cbits)
num_circuit = len(D_train)
objective_func_vals = []
optimizer = COBYLA(maxiter=100)
backend = Aer.get_backend('statevector_simulator')

qc = QuantumCircuit(qr, trash_r, ref_r, aux_r, cr)
qr_bits = list(range(num_qbits))
trash_bits = list(range(num_qbits - num_trash, num_qbits))
ref_bits = list(range(num_qbits, num_qbits + num_trash))
aux_bits = list(range(num_qbits + num_trash, num_qbits + num_trash + num_aux))

encoder_u = encoder(num_qbits, reps=2, name='u1')
encoder_v = encoder(num_qbits, reps=2, name='v1')
num_u_theta = encoder_u.num_parameters
num_v_theta = encoder_v.num_parameters

# 1: Initialize θ randomly.
init_theta_u = list(np.random.rand(encoder_u.num_parameters))
init_theta_v = list(np.random.rand(encoder_v.num_parameters))
init_theta = []
init_theta.extend(init_theta_u)
init_theta.extend(init_theta_v)
pars_u = init_theta[:num_u_theta]
pars_v = init_theta[num_u_theta:]
converged = False


# 2: while not converged or the iteration ITR is not satisfied do
while not converged or n_iter<200:
    # 3: Initialize loss L(θit) = 0;
    loss = 0
    # 4: for each quantum circuit E_i in D_train do
    for i_circuit in D_train:
        # 5: Apply encoders U(θit) and V(θit) on the training data Ei
        #    to get the channel Π = V(θit) ◦ E_i ◦ U(θit);
        # 6: Apply Π on a composite system A′ and C1 with states ωA′ and ϕ+ C1;
        pi_channel = encode_qc(num_qbits, num_trash, i_circuit, encoder_u, encoder_v, pars_u, pars_v)
        qc.compose(pi_channel, qubits=qr_bits, inplace=True)
        # 7: Apply a swap test on the “trash” system C2 and ancila system C1′ to estimate the
        #    fidelity between state of C2 and ϕ+ C1 and calculate Li 3(θit);
        qc = swap_test(qc, cr, ref_r, trash_r, aux_r)
        #partial_trace()
        shots = 1024
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        loss = counts['1']/shots

        # 8: Let L(θit)+ = n1Li 3(θit);
        loss += loss()
        # 9: end for
    # 10: Update parameters θit+1 of L(·) using classical optimizer;
    opt_result = optimizer.minimize(fun=loss_fun, x0=init_theta)
    theta_opt = opt_result.x
    # 11: end while
    exit()
#return theta_opt
# 12: Output the optimal parameters θOP T ;



