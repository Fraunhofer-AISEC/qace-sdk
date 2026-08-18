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
"""Preimage search using quantum amplitude amplification.

This module provides a quantum algorithm for finding preimages of a given
image under a vectorial Boolean function. It leverages Grover's diffusion
operator and quantum amplitude amplification to efficiently search for
inputs that map to a specified output.
"""

from dataclasses import dataclass, field

from qiskit.circuit import QuantumCircuit

from qace.algorithm import QuantumAlgorithm
from qace.execution import CircuitExecutor
from qace.vbf import VectorialBooleanFunction, int_to_fixed_bin_str
from qace.algorithm import (
    QAAConfig,
    QAAResult,
    QuantumAmplitudeAmplification,
)


@dataclass
class PreimageResult(QAAResult):
    """Result of a preimage search algorithm execution.

    Attributes:
        preimages: A list of integer preimage values found during the search.
    """

    preimages: list[int] = field(default_factory=list)


@dataclass
class PreimageConfig:
    """Configuration for the preimage search algorithm.

    Attributes:
        vbf: The vectorial Boolean function to find preimages for.
        vbf_image: The target image value whose preimage is sought.
        success_prob: Optional desired success probability for the algorithm.
        seed: Optional random seed for reproducibility.
        base: Optional base parameter for amplitude amplification scheduling.
    """

    vbf: VectorialBooleanFunction
    vbf_image: int
    success_prob: float | None = None
    seed: int | None = None
    base: float | None = None


class Preimage(QuantumAlgorithm):
    """Abstract base class for all preimage algorithms."""

    def __init__(
        self,
        executor: CircuitExecutor,
        preimage_config: PreimageConfig,
    ):
        """Initializes the Preimage algorithm instance.

        Args:
            executor: The circuit executor used to run quantum circuits.
            preimage_config: Configuration specifying the vectorial Boolean
                function and target image for the preimage search.
        """
        super().__init__(executor)
        self._preimage_config = preimage_config

    def _build_grover_circuit(self, unitary: QuantumCircuit) -> QuantumCircuit:
        """Builds the Grover diffusion operator circuit.

        Constructs the oracle and diffusion components of Grover's algorithm
        tailored to the configured vectorial Boolean function and target image.

        Args:
            unitary: The quantum circuit representing the vectorial Boolean
              function.

        Returns:
            A QuantumCircuit implementing one Grover iteration.
        """
        qubit_count = unitary.num_qubits
        aux_qubit_index = qubit_count - 1
        m = self._preimage_config.vbf.m
        n = self._preimage_config.vbf.n

        grover_circuit = QuantumCircuit(qubit_count)

        # Negate desired mask pairs
        grover_circuit.compose(
            build_phase_flip_x_circuit(n, self._preimage_config.vbf_image),
            qubits=range(
                m, m + n + 1
            ),  # the phase flip operates on the output of the vbf and the auxiliary
            inplace=True,
        )
        # `unitary^\dagger`
        grover_circuit.compose(
            unitary.inverse(), qubits=range(qubit_count), inplace=True
        )
        # Negate 0
        grover_circuit.compose(
            build_phase_flip_x_circuit(qubit_count - 1, 0),
            range(qubit_count),
            inplace=True,
        )
        # `unitary`
        grover_circuit.compose(unitary, qubits=range(qubit_count), inplace=True)
        # Global -1
        grover_circuit.x(aux_qubit_index)
        grover_circuit.z(aux_qubit_index)
        grover_circuit.x(aux_qubit_index)

        return grover_circuit

    def run(self) -> PreimageResult:
        """Executes the preimage search algorithm.

        Constructs the necessary quantum circuits and delegates execution to
        the quantum amplitude amplification subroutine.

        Returns:
            A PreimageResult containing the measurement outcome and iteration count.
        """
        m = self._preimage_config.vbf.m
        n = self._preimage_config.vbf.n
        unitary = self._preimage_config.vbf.to_circuit()
        qubit_count = unitary.num_qubits + 1
        unitary_extended = QuantumCircuit(qubit_count)
        unitary_extended.h(range(m))
        unitary_extended.compose(
            unitary, qubits=range(unitary.num_qubits), inplace=True
        )
        grover_circuit = self._build_grover_circuit(unitary_extended)

        def verify(measured_value):
            # we use n + 1 here to shift the measured value, because the output uses an auxiliary qubit
            x = (measured_value >> (n + 1)) & ((1 << m) - 1)
            # shift away the auxiliary, then get the output bits
            y = (measured_value >> 1) & ((1 << n) - 1)

            return (
                self._preimage_config.vbf.eval(x) == y
                and self._preimage_config.vbf_image == y
            )

        qaa_config = QAAConfig(
            unitary=unitary_extended,
            grover_circuit=grover_circuit,
            qubit_count=qubit_count,
            verify=verify,
            success_prob=self._preimage_config.success_prob,
            base=self._preimage_config.base,
            seed=self._preimage_config.seed,
        )

        qaa = QuantumAmplitudeAmplification(
            executor=self._executor,
            qaa_config=qaa_config,
        )
        result = qaa.run()

        return PreimageResult(
            execution_result=result.execution_result,
            metadata=result.metadata,
            verified_results=result.verified_results,
            iterations=result.iterations,
            preimages=[
                res >> (self._preimage_config.vbf.n + 1)
                for res in result.verified_results
            ],
        )


def build_phase_flip_x_circuit(n: int, x: int) -> QuantumCircuit:
    """Builds a phase flip circuit that flips the phase of a target state.

    Constructs a quantum circuit that applies a phase flip to the
    computational basis state specified by the integer x.

    Args:
        n: The number of control qubits (excluding the auxiliary qubit).
        x: The integer representation of the target basis state to phase flip.

    Returns:
        A QuantumCircuit of n + 1 qubits that applies a phase flip to state x.
    """
    flip_label = int_to_fixed_bin_str(x, n)[
        ::-1
    ]  # we use the convention that q0 is MSB
    phase_flip_x_circuit = QuantumCircuit(n + 1)
    phase_flip_x_circuit.x(n)
    phase_flip_x_circuit.h(n)
    phase_flip_x_circuit.mcx(
        phase_flip_x_circuit.qubits[0:n],
        phase_flip_x_circuit.qubits[n],
        ctrl_state=flip_label,
    )
    phase_flip_x_circuit.h(n)
    phase_flip_x_circuit.x(n)
    return phase_flip_x_circuit
