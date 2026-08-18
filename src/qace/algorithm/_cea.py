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
"""Module implementing the Correlation Extraction Algorithm (CEA).

This module provides the CorrelationExtraction quantum algorithm, which
extracts correlated mask pairs (alpha, beta) from a vectorial Boolean
function by constructing and sampling from the U_CEA^f circuit.
"""

from dataclasses import dataclass, field

from qiskit.circuit import QuantumCircuit

from qace.algorithm import QuantumAlgorithm, AlgorithmResult
from qace.execution import CircuitExecutor
from qace.vbf import VectorialBooleanFunction, measure_all_msb0


@dataclass
class CorrelationExtractionResult(AlgorithmResult):
    """Result of a CEA execution containing extracted mask pairs."""

    mask_pairs: dict[tuple[int, int], int] = field(default_factory=dict)


class CorrelationExtraction(QuantumAlgorithm):
    """Correlation Extraction Algorithm (CEA).

    Extracts correlated mask pairs (alpha, beta) from a vectorial Boolean
    function by constructing and sampling from the U_CEA^f circuit.

    The circuit applies:
        1. Hadamard on all input qubits
        2. The function oracle U_f
        3. Hadamard on all qubits (input + output)

    Measurement yields mask pairs (alpha, beta) where the correlation
    between alpha and beta reveals structural information about f.
    """

    def __init__(self, executor: CircuitExecutor, vbf: VectorialBooleanFunction):
        """Initializes the CEA with an executor and a vectorial Boolean function.

        Args:
            executor: The CircuitExecutor instance used to run the circuit.
            vbf: The VectorialBooleanFunction to analyze.
        """
        super().__init__(executor)
        self.vbf = vbf

    def build_circuit(self, measure=True) -> QuantumCircuit:
        """Build the full CEA circuit including measurement.

        Constructs U_CEA^f followed by measurement on all qubits.
        Qubits are measured in reverse order to ease mask reconstruction.

        Returns:
            A measured QuantumCircuit ready for execution.
        """
        m = self.vbf.m
        n = self.vbf.n
        num_qubits = m + n

        # Build the core CEA unitary
        cea_unitary = self.build_cea_unitary()

        # Wrap in a measured circuit
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(cea_unitary, qubits=range(num_qubits), inplace=True)
        if measure:
            measure_all_msb0(circuit)

        return circuit

    def build_cea_unitary(self) -> QuantumCircuit:
        """Build the CEA unitary U_CEA^f (without measurement).

        Applies H^m, then U_f, then H^(m+n).

        Returns:
            The unitary CEA circuit.
        """
        m = self.vbf.m
        n = self.vbf.n
        num_qubits = m + n

        f_circuit = self.vbf.to_circuit()

        cea_circuit = QuantumCircuit(num_qubits)
        cea_circuit.h(range(m))
        cea_circuit.compose(f_circuit, qubits=range(num_qubits), inplace=True)
        cea_circuit.h(range(num_qubits))

        return cea_circuit

    def run(self) -> CorrelationExtractionResult:
        """Execute the CEA and extract mask pairs from measurement results.

        Returns:
            CorrelationExtractionResult with extracted (alpha, beta) pairs.
        """
        circuit = self.build_circuit()
        execution_result = self._executor.execute(circuit)

        mask_pairs = self._counts_to_mask_pairs(execution_result.counts)

        return CorrelationExtractionResult(
            execution_result=execution_result,
            metadata={},
            mask_pairs=mask_pairs,
        )

    def _counts_to_mask_pairs(
        self, counts: dict[str, int]
    ) -> dict[tuple[int, int], int]:
        """Convert raw measurement counts to (alpha, beta) -> count mapping.

        Args:
            counts: Measurement results as bitstring -> count mapping.

        Returns:
            Dictionary mapping (alpha, beta) tuples to their counts.
        """
        m = self.vbf.m
        n = self.vbf.n
        mask_pairs: dict[tuple[int, int], int] = {}

        for bitstring, count in counts.items():
            value = (
                int(bitstring, 16) if bitstring.startswith("0x") else int(bitstring, 2)
            )

            alpha = (value >> n) & ((1 << m) - 1)
            beta = value & ((1 << n) - 1)

            pair = (alpha, beta)
            mask_pairs[pair] = count

        return mask_pairs
