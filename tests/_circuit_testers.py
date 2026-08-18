# Copyright 2026 Fraunhofer AISEC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper utilities for testing quantum circuits.

This module provides auxiliary test functions that validate the behavior of
quantum circuits either by performing computational-basis measurements or by
inspecting the resulting statevectors. Both helpers rely on a user-supplied
predicate to determine whether the observed outcome matches the expected
behavior for every canonical basis input.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qace.execution import AerExecutor
from qace.vbf import measure_msb0


def aux_test_circuit_by_measurements_with_predicate(circuit, verify):
    """Tests a quantum circuit by measuring outcomes for all canonical inputs.

    For a given quantum circuit, this function performs a measurement in the
    canonical basis for every result of the circuit, when applied to an initial
    canonical state. The result is plugged into the verification function
    passed to assert whether the circuit behaves as expected.

    Args:
        circuit: QuantumCircuit to test.
        verify: Verification function of signature int, int -> bool with
            arguments x, reg, where x represents the canonical state
            initialized and reg represents the canonical state measured.
    """

    qubit_count = len(circuit.qubits)
    executor = AerExecutor.default()
    for x in range(1 << qubit_count):
        dummy = QuantumCircuit(qubit_count, qubit_count)
        dummy.initialize(x, qubits=list(reversed(range(qubit_count))))
        dummy.compose(circuit, qubits=list(range(qubit_count)), inplace=True)
        measure_msb0(dummy, range(qubit_count), range(qubit_count))
        execution_results = executor.execute(dummy)
        result, *_ = execution_results.counts.keys()
        reg = int(result, 2)
        assert verify(x, reg)


def aux_test_circuit_by_statevectors_with_predicate(circuit, verify):
    """Tests a quantum circuit by inspecting statevectors for all canonical inputs.

    For a given quantum circuit, this function computes the resulting
    statevector for every canonical basis input state and passes it to the
    verification function to assert whether the circuit behaves as expected.

    Args:
        circuit: QuantumCircuit to test.
        verify: Verification function of signature int, Statevector -> bool
            with arguments x, statevector, where x represents the canonical
            state initialized and statevector represents the resulting
            statevector after applying the circuit.
    """
    qubit_count = len(circuit.qubits)
    for x in range(1 << qubit_count):
        dummy = QuantumCircuit(qubit_count)
        dummy.initialize(x, qubits=list(reversed(range(qubit_count))))
        dummy.compose(circuit, qubits=list(range(qubit_count)), inplace=True)

        statevector = Statevector(dummy)
        assert verify(x, statevector)
