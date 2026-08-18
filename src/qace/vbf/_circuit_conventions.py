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
"""Circuit utilities enforcing the project-wide MSB₀ qubit convention.

Throughout this project, qubit index 0 represents the most significant bit.
Qiskit natively uses LSB₀ (qubit index 0 = least significant bit).
These helpers translate between the two conventions automatically.

WARNING: Do not use bare `circuit.mcx(..., ctrl_state=...)` calls elsewhere
in the codebase. Always use `mcx_msb0()` to avoid bit-ordering bugs.
"""

from collections.abc import Sequence

from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.quantumcircuit import QubitSpecifier, ClbitSpecifier


def mcx_msb0(
    circuit: QuantumCircuit,
    control_qubits: Sequence[QubitSpecifier],
    target_qubit: QubitSpecifier,
    ctrl_state: str | int | None = None,
):
    """Multi-controlled X gate using MSB₀ convention (qubit index 0 = MSB).

    Translates from the project's MSB₀ convention to Qiskit's internal
    LSB₀ convention automatically.

    Args:
        circuit: The quantum circuit to append the gate to.
        control_qubits: Control qubit indices in MSB₀ order.
        target_qubit: Target qubit index.
        ctrl_state: Value with MSB₀ order of the control state to trigger on.
    """
    circuit.mcx(
        control_qubits[::-1],
        target_qubit,
        ctrl_state=ctrl_state,
    )


def measure_msb0(
    circuit: QuantumCircuit, qubits: QubitSpecifier, clbits: ClbitSpecifier
):
    """Measurement using MSB₀ convention (qubit 0 maps to the most significant classical bit).

    Wraps Qiskit's `measure` but ensures that qubit index 0 maps to
    the most significant bit in the resulting bitstring.

    Args:
        circuit: The quantum circuit to append measurements to.
        qubits: Qubit indices to measure (in MSB₀ order).
        clbits: Classical bit indices to store results.
    """
    circuit.measure(qubits[::-1], clbits)


def measure_all_msb0(
    circuit: QuantumCircuit,
    inplace: bool = True,
    add_bits: bool = True,
) -> QuantumCircuit | None:
    """Measures all qubits using MSB₀ convention (qubit 0 → most significant classical bit).

    Wraps Qiskit's `measure_all()` but ensures that qubit index 0 maps to
    the most significant bit in the resulting bitstring.

    Args:
        circuit: The quantum circuit to append measurements to.
        inplace: If True, adds measurements to the existing circuit.
            If False, returns a new circuit with measurements.
        add_bits: If True, adds new classical bits in a ClassicalRegister
            to store the measurements. If False, stores results in
            already existing classical bits (qubit n → classical bit
            (num_qubits - 1 - n) to maintain MSB₀ ordering).

    Returns:
        None if inplace=True, otherwise a new circuit with measurements.

    Raises:
        CircuitError: If add_bits=False but there are not enough classical bits.
    """
    n = circuit.num_qubits

    if not inplace:
        circuit = circuit.copy()

    if add_bits:
        circuit.add_register(ClassicalRegister(n))

    circuit.measure(range(n)[::-1], range(n))

    if not inplace:
        return circuit
    else:
        return None
