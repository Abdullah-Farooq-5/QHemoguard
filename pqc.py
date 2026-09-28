from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

def build_U(num_features, x, omega):
    """Builds the diagonal phase unitary U for num_features computational qubits."""
    U = QuantumCircuit(num_features, name='U')

    for state in range(2**num_features):
        bits = format(state, f'0{num_features}b')
        flip_qubits = [i for i, b in enumerate(bits) if b == '0']

        for q in flip_qubits:
            U.x(q)

        phase = sum(x[i] * omega[2*i + int(bits[i])] for i in range(num_features))

        if num_features == 1:
            U.p(phase, 0)
        else:
            U.mcp(phase, list(range(num_features - 1)), num_features - 1)

        for q in flip_qubits:
            U.x(q)

    return U

def build_full_pqc(num_features):
    """Returns the complete PQC (classifier + computational qubits, Hadamard test wrapper)."""
    x = ParameterVector('x', num_features)
    omega = ParameterVector('w', 2*num_features)

    U = build_U(num_features, x, omega)
    U = transpile(U)
    CU_gate = U.control(1)

    q_class = QuantumRegister(1, name='Q')
    q_comp  = QuantumRegister(num_features, name='q')
    c       = ClassicalRegister(1, name='c')
    qc      = QuantumCircuit(q_class, q_comp, c, name='qc')

    qc.h(q_class[0])
    for i in range(num_features):
        qc.h(q_comp[i])
    qc.append(CU_gate, [q_class[0]] + [q_comp[i] for i in range(num_features)])
    qc.h(q_class[0])
    qc.measure(q_class[0], c[0])

    return qc, x, omega