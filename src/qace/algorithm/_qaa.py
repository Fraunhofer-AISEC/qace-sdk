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
"""Module implementing the Quantum Amplitude Amplification (QAA) algorithm.

Provides the QAA algorithm following "Quantum Amplitude Amplification and
Estimation" by Brassard et al., including circuit construction, iterative
measurement, and both known and unknown success probability variants.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from math import asin, ceil, floor, pi, sqrt

from qiskit.circuit import QuantumCircuit

from qace.algorithm import QuantumAlgorithm, AlgorithmResult
from qace.execution import CircuitExecutor, ExecutionResult
from qace.vbf import measure_all_msb0


@dataclass
class QAAConfig:
    """Circuit configuration for Quantum Amplitude Amplification.

    Attributes:
        unitary: The initial unitary quantum circuit to amplify.
        grover_circuit: The Grover operator circuit used for amplification.
        qubit_count: The total number of qubits in the circuits.
        verify: A callable that takes an integer measurement result and
            returns True if it is a valid solution.
        success_prob: If known, the probability of measuring a valid
            result from the initial unitary applied to |0>. Enables
            precomputation of the optimal Grover iteration count.
            If None, the exponential search strategy is used.
        base: Base for exponential increase of the search space per
            iteration. If None, a random value in (1, 2) is chosen.
        seed: Seed for classical random choices in the algorithm
            (base selection and iteration count sampling).
    """

    unitary: QuantumCircuit
    grover_circuit: QuantumCircuit
    qubit_count: int
    verify: Callable[[int], bool]
    success_prob: float | None = None
    base: float | None = None
    seed: int | None = None

    def __post_init__(self):
        """Validates configuration constraints.

        Raises:
            ValueError: If base is not in (1, 2).
        """
        if self.base is not None and not (1.0 < self.base < 2.0):
            raise ValueError(
                f"base must be in the open interval (1, 2), got {self.base}."
            )


@dataclass
class QAAResult(AlgorithmResult):
    """Result of a QAA execution containing verified measurement results.

    Attributes:
        verified_results: List of integer measurement results that passed
            the verification function.
        iterations: Number of iterations the algorithm performed before
            finding a verified result.
    """

    verified_results: list[int] = field(default_factory=list)
    iterations: int = 0


class QuantumAmplitudeAmplification(QuantumAlgorithm):
    """Quantum Amplitude Amplification (QAA) algorithm.

    Implements the QAA algorithm from "Quantum Amplitude Amplification and
    Estimation" by Brassard et al. Supports both the known success probability
    case (deterministic repetition count) and the unknown success probability
    case (exponential search).

    The algorithm iteratively builds circuits with increasing numbers of
    Grover operator applications and measures until a verified result is found.

    Note:
        For correct behavior, the injected executor should NOT use a fixed
        `seed_simulator` with `shots=1`, as this would produce identical
        measurement outcomes across iterations, preventing convergence.
        Either omit the seed or use `shots > 1`. For deterministic executions of QAA
        use the IterativeExecutor as executor.
    """

    def __init__(
        self,
        executor: CircuitExecutor,
        qaa_config: QAAConfig,
    ):
        """Initializes the QAA algorithm.

        Args:
            executor: The CircuitExecutor instance used to run circuits.
            qaa_config: Configuration containing the unitary, Grover circuit,
                and qubit count.
        """
        super().__init__(executor)
        self._qaa_config = qaa_config

    def build_circuit(self, repetitions: int = 0) -> QuantumCircuit:
        """Builds a QAA circuit with the given number of Grover operator applications.

        Constructs a circuit that applies the initial unitary followed by
        `repetitions` applications of the Grover operator, then measures
        all qubits.

        Args:
            repetitions: Number of times to apply the Grover operator.

        Returns:
            A measured QuantumCircuit ready for execution.
        """
        unitary = self._qaa_config.unitary
        grover_circuit = self._qaa_config.grover_circuit
        qubit_count = self._qaa_config.qubit_count

        circuit = QuantumCircuit(qubit_count)
        circuit.compose(unitary, qubits=range(qubit_count), inplace=True)
        for _ in range(repetitions):
            circuit.compose(grover_circuit, qubits=range(qubit_count), inplace=True)
        measure_all_msb0(circuit)

        return circuit

    def run(self) -> QAAResult:
        """Executes the QAA algorithm.

        If success_prob is known, computes the optimal number of Grover
        iterations and measures until a verified result is found.
        If unknown, uses exponential search with random iteration counts.

        Returns:
            QAAResult containing verified measurement results and metadata.

        Note:
            This method may run indefinitely if no valid solution exists
            for the given verification function.
        """
        # Save the rng state to make it independent of the seed used in this class

        state_backup = random.getstate()
        random.seed(self._qaa_config.seed)

        if self._qaa_config.success_prob is None:
            verified_results, execution_result, iterations = (
                self._run_unknown_probability()
            )
        else:
            verified_results, execution_result, iterations = (
                self._run_known_probability()
            )

        # Restore the rng state
        random.setstate(state_backup)

        return QAAResult(
            execution_result=execution_result,
            metadata={"iterations": iterations},
            verified_results=verified_results,
            iterations=iterations,
        )

    def _run_known_probability(self) -> tuple[list[int], ExecutionResult, int]:
        """Runs QAA with known success probability.

        Computes the optimal iteration count from the known probability
        and repeatedly measures until a verified result is found.

        Returns:
            A tuple (verified_results, last_execution_result, iteration_count),
            where verified_results is the list of valid solutions,
            last_execution_result is the final ExecutionResult, and
            iteration_count is the number of measurement rounds performed.
        """
        aa_iter_count = floor(pi / (4 * asin(sqrt(self._qaa_config.success_prob))))
        iterations = 0

        while True:
            iterations += 1
            circuit = self.build_circuit(aa_iter_count)
            execution_result = self._executor.execute(circuit)
            results = self._counts_to_int_results(execution_result.counts)

            verified_results = [r for r in results if self._qaa_config.verify(r)]
            if len(verified_results) > 0:
                return verified_results, execution_result, iterations

    def _run_unknown_probability(self) -> tuple[list[int], ExecutionResult, int]:
        """Runs QAA with unknown success probability using exponential search.

        Chooses a random base c in (1, 2) (or uses the configured base),
        then exponentially increases the search space per iteration.

        Returns:
            A tuple (verified_results, last_execution_result, iteration_count),
            where verified_results is the list of valid solutions,
            last_execution_result is the final ExecutionResult, and
            iteration_count is the number of exponential search rounds performed.
        """
        if self._qaa_config.base is None:
            while True:
                c = random.uniform(1.0, 2.0)
                if c != 1.0 and c != 2.0:
                    break
        else:
            c = self._qaa_config.base

        l = 0
        while True:
            l += 1
            M = ceil(c**l)

            # Initial measurement (0 Grover iterations)
            circuit = self.build_circuit(0)
            execution_result = self._executor.execute(circuit)
            results = self._counts_to_int_results(execution_result.counts)

            verified_results = [r for r in results if self._qaa_config.verify(r)]
            if len(verified_results) > 0:
                return verified_results, execution_result, l

            # Amplified measurement (random j in [1, M] Grover iterations)
            j = random.randint(1, M)
            circuit = self.build_circuit(j)
            execution_result = self._executor.execute(circuit)
            results = self._counts_to_int_results(execution_result.counts)

            verified_results = [r for r in results if self._qaa_config.verify(r)]
            if len(verified_results) > 0:
                return verified_results, execution_result, l

    @staticmethod
    def _counts_to_int_results(counts: dict[str, int]) -> list[int]:
        """Converts measurement counts to a list of unique integer results.

        Args:
            counts: Measurement results as bitstring-to-count mapping.

        Returns:
            List of unique integer values observed in the measurements.
        """
        results = []
        for bitstring in counts.keys():
            if bitstring.startswith("0x"):
                value = int(bitstring, 16)
            else:
                value = int(bitstring, 2)
            results.append(value)
        return results
