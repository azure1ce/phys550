from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector, state_fidelity, DensityMatrix, partial_trace
from qiskit.visualization import plot_state_city
import numpy as np
import matplotlib.pyplot as plt
import json

# --- Setup IBM Backend ---
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# --- Bell state preparation ---
def create_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# --- Measurement basis transformation ---
def apply_pauli_measurement(qc, pauli_string):
    for i, p in enumerate(pauli_string):
        if p == 'x':
            qc.h(i)
        elif p == 'y':
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()

# --- Generate all tomography circuits ---
def generate_tomography_circuits(pauli_pairs):
    circuits = []
    metadata = []
    for pauli1, pauli2 in pauli_pairs:
        qc = create_bell_state()
        apply_pauli_measurement(qc, pauli1 + pauli2)
        qc.name = f"{pauli1}_{pauli2}"
        circuits.append(qc)
        metadata.append((pauli1, pauli2))
    return circuits, metadata

# --- Submit circuits and save raw results ---
def run_ibm_sampler(circuits, backend, shots=8192):
    sampler = Sampler(mode=backend)
    transpiled = transpile(circuits, backend=backend)
    job = sampler.run(transpiled, shots=shots)  # ✅ 不加中括号
    result = job.result()
    raw_counts = [r.data.meas.get_counts() for r in result]
    return raw_counts

# --- Process counts to extract expectation values ---
def process_counts_to_expectation(raw_counts, metadata):
    expectations = {}
    for counts, (pauli1, pauli2) in zip(raw_counts, metadata):
        total = sum(counts.values())
        exp = 0
        for outcome, count in counts.items():
            bits = outcome.zfill(2)[-2:]
            bit0 = int(bits[0])
            bit1 = int(bits[1])
            sign = 1 if bit0 == bit1 else -1
            exp += sign * count
        expectations[(pauli1, pauli2)] = exp / total
    return expectations

# --- Reconstruct density matrix ---
def reconstruct_density_matrix(expectations):
    rho = np.eye(4, dtype=complex) / 4
    for (i, j), v in expectations.items():
        label = f"{i}{j}".upper()  # e.g., "xx" -> "XX"
        pauli_matrix = Pauli(label).to_matrix()
        rho += 0.5 * v * pauli_matrix / 2
    return rho
    
import pandas as pd
# --- Save Data Sheet ---
def save_raw_counts_to_csv(raw_counts, metadata, filename="tomo_raw_counts.csv"):
    """
    Save raw_counts and metadata to a CSV file with bitstrings always shown as 2-bit string.
    """
    all_rows = []
    for counts, (p1, p2) in zip(raw_counts, metadata):
        for bitstring, count in counts.items():
            padded = str(bitstring).zfill(2)  # ensure it's a 2-character string
            all_rows.append({
                "pauli1": p1,
                "pauli2": p2,
                "bitstring": f'"{padded}"',  # force Excel to interpret as text
                "count": count
            })

    df = pd.DataFrame(all_rows, dtype=str)
    df.to_csv(filename, index=False, quoting=1)  # quoting=1 = QUOTE_ALL
    print(f"✅ CSV已保存，bitstring作为完整字符串写入: {filename}")

# --- Main execution ---
pauli_labels = ['x', 'y', 'z']
pauli_pairs = [(i, j) for i in pauli_labels for j in pauli_labels]
circuits, metadata = generate_tomography_circuits(pauli_pairs)
raw_counts = run_ibm_sampler(circuits, backend)
expectations = process_counts_to_expectation(raw_counts, metadata)
rho = reconstruct_density_matrix(expectations)

# --- Visualization ---
fig = plot_state_city(DensityMatrix(rho), title="Reconstructed ρ")
fig.savefig("reconstructed_density_matrix.png")
print("图像已保存为 reconstructed_density_matrix.png")

# --- Data Store ---
save_raw_counts_to_csv(raw_counts, metadata, filename="my_tomo_counts.csv")
fig  # 显示图像