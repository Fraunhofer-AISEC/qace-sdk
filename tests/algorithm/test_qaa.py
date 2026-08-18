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
"""Tests for the Quantum Amplitude Amplification (QAA) module."""

import pytest
from qiskit.circuit import QuantumCircuit
from beartype.roar import BeartypeCallHintParamViolation

from qace.algorithm import (
    AlgorithmResult,
    QAAConfig,
    QAAResult,
    QuantumAmplitudeAmplification,
)
from qace.execution import (
    AerExecutor,
    ExecutionConfig,
    ExecutionResult,
    IterativeExecutor,
)

# === Helpers ===


def _build_2qubit_grover_for_target_11():
    """Build unitary and Grover operator for finding |11⟩ (=3) in 2 qubits.

    With success_prob=1/4, the optimal iteration count is 1,
    and after 1 Grover iteration the measurement probability for |11⟩ is 1.0.
    """
    unitary = QuantumCircuit(2)
    unitary.h([0, 1])

    grover = QuantumCircuit(2)
    # Oracle: phase flip on |11⟩
    grover.cz(0, 1)

    grover.h([0, 1])  # ─┐
    grover.z([0, 1])  # │Diffusion: H (2|0><0| - I) H = 2|s><s| - I
    grover.cz(0, 1)  # │
    grover.h([0, 1])  # ─┘

    return unitary, grover


# === Fixtures ===


@pytest.fixture
def grover_2q():
    """Unitary and Grover circuit for 2-qubit search targeting |11⟩."""
    return _build_2qubit_grover_for_target_11()


@pytest.fixture
def verify_is_3():
    """Verification function accepting only integer 3."""
    return lambda x: x == 3


@pytest.fixture
def qaa_config_known(grover_2q, verify_is_3):
    """QAAConfig with known success probability 1/4."""
    unitary, grover = grover_2q
    return QAAConfig(
        unitary=unitary,
        grover_circuit=grover,
        qubit_count=2,
        verify=verify_is_3,
        success_prob=0.25,
        seed=42,
    )


@pytest.fixture
def qaa_config_unknown(grover_2q, verify_is_3):
    """QAAConfig with unknown success probability."""
    unitary, grover = grover_2q
    return QAAConfig(
        unitary=unitary,
        grover_circuit=grover,
        qubit_count=2,
        verify=verify_is_3,
        success_prob=None,
        base=1.5,
        seed=42,
    )


@pytest.fixture
def aer_executor():
    """Seeded AerExecutor with 1024 shots."""
    config = ExecutionConfig(
        backend_options={"seed_simulator": 42},
        run_options={"shots": 1024},
        transpiler_options={"seed_transpiler": 42},
        skip_transpilation=True,
    )
    return AerExecutor(config=config)


@pytest.fixture
def iterative_executor():
    """IterativeExecutor with single shot for multi-call determinism."""
    config = ExecutionConfig(run_options={"shots": 1}, skip_transpilation=True)
    return IterativeExecutor(AerExecutor(config), base_seed=100)


@pytest.fixture
def qaa_known(aer_executor, qaa_config_known):
    """QAA instance with known success probability."""
    return QuantumAmplitudeAmplification(
        executor=aer_executor,
        qaa_config=qaa_config_known,
    )


@pytest.fixture
def qaa_unknown(aer_executor, qaa_config_unknown):
    """QAA instance with unknown success probability."""
    return QuantumAmplitudeAmplification(
        executor=aer_executor,
        qaa_config=qaa_config_unknown,
    )


# === Tests: QAAConfig initialization ===


class TestQAAConfig:

    def test_all_fields_stored(self, grover_2q, verify_is_3):
        """All fields are stored correctly on initialization."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=0.25,
            base=1.5,
            seed=7,
        )
        assert config.unitary is unitary
        assert config.grover_circuit is grover
        assert config.qubit_count == 2
        assert config.verify is verify_is_3
        assert config.success_prob == 0.25
        assert config.base == 1.5
        assert config.seed == 7

    def test_defaults_none_for_optional_fields(self, grover_2q, verify_is_3):
        """Optional fields default to None."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
        )
        assert config.success_prob is None
        assert config.base is None
        assert config.seed is None


# === Tests: QAAConfig base validation ===


class TestQAAConfigBaseValidation:

    def test_base_none_is_valid(self, grover_2q, verify_is_3):
        """base=None does not raise."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            base=None,
        )
        assert config.base is None

    def test_base_1_5_is_valid(self, grover_2q, verify_is_3):
        """base=1.5 (within (1, 2)) does not raise."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            base=1.5,
        )
        assert config.base == 1.5

    def test_base_exactly_1_raises(self, grover_2q, verify_is_3):
        """base=1.0 raises ValueError (not in open interval)."""
        unitary, grover = grover_2q
        with pytest.raises(ValueError, match="base must be in the open interval"):
            QAAConfig(
                unitary=unitary,
                grover_circuit=grover,
                qubit_count=2,
                verify=verify_is_3,
                base=1.0,
            )

    def test_base_exactly_2_raises(self, grover_2q, verify_is_3):
        """base=2.0 raises ValueError (not in open interval)."""
        unitary, grover = grover_2q
        with pytest.raises(ValueError, match="base must be in the open interval"):
            QAAConfig(
                unitary=unitary,
                grover_circuit=grover,
                qubit_count=2,
                verify=verify_is_3,
                base=2.0,
            )

    def test_base_below_1_raises(self, grover_2q, verify_is_3):
        """base=0.5 raises ValueError."""
        unitary, grover = grover_2q
        with pytest.raises(ValueError, match="base must be in the open interval"):
            QAAConfig(
                unitary=unitary,
                grover_circuit=grover,
                qubit_count=2,
                verify=verify_is_3,
                base=0.5,
            )

    def test_base_above_2_raises(self, grover_2q, verify_is_3):
        """base=2.5 raises ValueError."""
        unitary, grover = grover_2q
        with pytest.raises(ValueError, match="base must be in the open interval"):
            QAAConfig(
                unitary=unitary,
                grover_circuit=grover,
                qubit_count=2,
                verify=verify_is_3,
                base=2.5,
            )

    def test_base_just_above_1_is_valid(self, grover_2q, verify_is_3):
        """base=1.01 is valid."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            base=1.01,
        )
        assert config.base == 1.01

    def test_base_just_below_2_is_valid(self, grover_2q, verify_is_3):
        """base=1.99 is valid."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            base=1.99,
        )
        assert config.base == 1.99


# === Tests: QAAResult dataclass ===


class TestQAAResult:

    def test_default_verified_results_empty(self):
        """Default verified_results is an empty list."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        result = QAAResult(execution_result=exec_result)
        assert result.verified_results == []

    def test_default_iterations_zero(self):
        """Default iterations is 0."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        result = QAAResult(execution_result=exec_result)
        assert result.iterations == 0

    def test_custom_verified_results_stored(self):
        """Custom verified_results are stored correctly."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        result = QAAResult(
            execution_result=exec_result,
            verified_results=[3, 7],
            iterations=5,
        )
        assert result.verified_results == [3, 7]
        assert result.iterations == 5

    def test_inherits_algorithm_result(self):
        """QAAResult is an AlgorithmResult."""
        exec_result = ExecutionResult(counts={"11": 100}, result=None, metadata={})
        result = QAAResult(
            execution_result=exec_result,
            metadata={"key": "value"},
        )
        assert isinstance(result, AlgorithmResult)
        assert result.execution_result is exec_result
        assert result.metadata == {"key": "value"}

    def test_verified_results_not_shared_between_instances(self):
        """Each instance has its own verified_results list."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        r1 = QAAResult(execution_result=exec_result)
        r2 = QAAResult(execution_result=exec_result)
        r1.verified_results.append(42)
        assert 42 not in r2.verified_results


# === Tests: QuantumAmplitudeAmplification initialization ===


class TestQuantumAmplitudeAmplificationInit:

    def test_executor_stored(self, aer_executor, qaa_config_known):
        """Executor is stored on the instance."""
        qaa = QuantumAmplitudeAmplification(
            executor=aer_executor,
            qaa_config=qaa_config_known,
        )
        assert qaa._executor is aer_executor

    def test_config_stored(self, aer_executor, qaa_config_known):
        """QAAConfig is stored on the instance."""
        qaa = QuantumAmplitudeAmplification(
            executor=aer_executor,
            qaa_config=qaa_config_known,
        )
        assert qaa._qaa_config is qaa_config_known


# === Tests: Type checking ===


class TestQuantumAmplitudeAmplificationTypeChecking:

    def test_none_executor_raises(self, qaa_config_known):
        """None executor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            QuantumAmplitudeAmplification(
                executor=None,
                qaa_config=qaa_config_known,
            )

    def test_wrong_type_executor_raises(self, qaa_config_known):
        """Non-CircuitExecutor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            QuantumAmplitudeAmplification(
                executor="not_an_executor",
                qaa_config=qaa_config_known,
            )

    def test_none_config_raises(self, aer_executor):
        """None config raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            QuantumAmplitudeAmplification(
                executor=aer_executor,
                qaa_config=None,
            )

    def test_wrong_type_config_raises(self, aer_executor):
        """Non-QAAConfig raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            QuantumAmplitudeAmplification(
                executor=aer_executor,
                qaa_config="not_a_config",
            )


# === Tests: build_circuit ===


class TestBuildCircuit:

    def test_correct_qubit_count(self, qaa_known):
        """Circuit has qubit_count qubits."""
        circuit = qaa_known.build_circuit(repetitions=0)
        assert circuit.num_qubits == 2

    def test_correct_clbit_count(self, qaa_known):
        """Circuit has qubit_count classical bits."""
        circuit = qaa_known.build_circuit(repetitions=0)
        assert circuit.num_clbits == 2

    def test_has_measurements(self, qaa_known):
        """Circuit contains measurement operations."""
        circuit = qaa_known.build_circuit(repetitions=0)
        op_names = [inst.operation.name for inst in circuit.data]
        assert "measure" in op_names

    def test_measurement_count_equals_qubit_count(self, qaa_known):
        """Circuit has exactly qubit_count measurements."""
        circuit = qaa_known.build_circuit(repetitions=0)
        measure_count = sum(
            1 for inst in circuit.data if inst.operation.name == "measure"
        )
        assert measure_count == 2

    def test_zero_repetitions_smaller_than_one_repetition(self, qaa_known):
        """Circuit with 0 repetitions is smaller than with 1 repetition."""
        circuit_0 = qaa_known.build_circuit(repetitions=0)
        circuit_1 = qaa_known.build_circuit(repetitions=1)
        assert circuit_0.size() < circuit_1.size()

    def test_more_repetitions_increases_depth(self, qaa_known):
        """More Grover repetitions increase circuit depth."""
        circuit_1 = qaa_known.build_circuit(repetitions=1)
        circuit_3 = qaa_known.build_circuit(repetitions=3)
        assert circuit_3.depth() > circuit_1.depth()

    def test_idempotent(self, qaa_known):
        """Multiple calls with same repetitions produce structurally equivalent circuits."""
        c1 = qaa_known.build_circuit(repetitions=2)
        c2 = qaa_known.build_circuit(repetitions=2)
        assert c1.num_qubits == c2.num_qubits
        assert c1.depth() == c2.depth()
        assert c1.size() == c2.size()

    def test_default_repetitions_is_zero(self, qaa_known):
        """Default repetitions is 0 (no Grover applications)."""
        circuit_default = qaa_known.build_circuit()
        circuit_zero = qaa_known.build_circuit(repetitions=0)
        assert circuit_default.depth() == circuit_zero.depth()
        assert circuit_default.size() == circuit_zero.size()


# === Tests: _counts_to_int_results ===


class TestCountsToIntResults:

    def test_empty_counts_returns_empty_list(self):
        """Empty counts dict returns empty list."""
        result = QuantumAmplitudeAmplification._counts_to_int_results({})
        assert result == []

    def test_single_binary_string(self):
        """Single binary bitstring is correctly converted."""
        result = QuantumAmplitudeAmplification._counts_to_int_results({"11": 100})
        assert result == [3]

    def test_single_hex_string(self):
        """Single hex bitstring with 0x prefix is correctly converted."""
        result = QuantumAmplitudeAmplification._counts_to_int_results({"0x3": 100})
        assert result == [3]

    def test_multiple_binary_strings(self):
        """Multiple binary bitstrings produce correct integer list."""
        counts = {"00": 50, "01": 30, "10": 15, "11": 5}
        result = QuantumAmplitudeAmplification._counts_to_int_results(counts)
        assert sorted(result) == [0, 1, 2, 3]

    def test_multiple_hex_strings(self):
        """Multiple hex bitstrings produce correct integer list."""
        counts = {"0x0": 50, "0xa": 30, "0xf": 20}
        result = QuantumAmplitudeAmplification._counts_to_int_results(counts)
        assert sorted(result) == [0, 10, 15]

    def test_returns_unique_values_only(self):
        """Each bitstring key appears exactly once regardless of count."""
        counts = {"101": 999}
        result = QuantumAmplitudeAmplification._counts_to_int_results(counts)
        assert result == [5]

    def test_all_zeros_binary(self):
        """All-zero binary string converts to 0."""
        result = QuantumAmplitudeAmplification._counts_to_int_results({"0000": 10})
        assert result == [0]

    def test_all_ones_binary(self):
        """All-one binary string converts correctly."""
        result = QuantumAmplitudeAmplification._counts_to_int_results({"1111": 10})
        assert result == [15]


# === Tests: run with known probability ===


class TestRunKnownProbability:

    def test_returns_qaa_result(self, qaa_known):
        """run() returns a QAAResult instance."""
        result = qaa_known.run()
        assert isinstance(result, QAAResult)

    def test_verified_results_contains_target(self, qaa_known):
        """Verified results contain the target value 3."""
        result = qaa_known.run()
        assert 3 in result.verified_results

    def test_verified_results_only_valid(self, qaa_known):
        """All verified results pass the verify function."""
        result = qaa_known.run()
        for r in result.verified_results:
            assert r == 3

    def test_iterations_is_positive(self, qaa_known):
        """At least 1 iteration was performed."""
        result = qaa_known.run()
        assert result.iterations >= 1

    def test_execution_result_stored(self, qaa_known):
        """ExecutionResult is stored in the result."""
        result = qaa_known.run()
        assert isinstance(result.execution_result, ExecutionResult)

    def test_execution_result_counts_not_empty(self, qaa_known):
        """ExecutionResult contains non-empty counts."""
        result = qaa_known.run()
        assert len(result.execution_result.counts) > 0

    def test_metadata_contains_iterations(self, qaa_known):
        """Metadata includes iteration count."""
        result = qaa_known.run()
        assert "iterations" in result.metadata
        assert result.metadata["iterations"] == result.iterations

    def test_perfect_grover_finds_target_in_one_iteration(self, qaa_known):
        """With probability-1 amplification, target is found in 1 iteration."""
        result = qaa_known.run()
        assert result.iterations == 1

    def test_success_prob_1_finds_immediately(
        self, aer_executor, grover_2q, verify_is_3
    ):
        """With success_prob=1.0 (trivial case), 0 Grover iterations suffice."""
        unitary = QuantumCircuit(2)
        unitary.x([0, 1])  # Prepare |11⟩ directly

        _, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=1.0,
            seed=42,
        )
        qaa = QuantumAmplitudeAmplification(executor=aer_executor, qaa_config=config)
        result = qaa.run()
        assert 3 in result.verified_results
        assert result.iterations == 1


# === Tests: run with unknown probability ===


class TestRunUnknownProbability:

    def test_returns_qaa_result(self, qaa_unknown):
        """run() with unknown probability returns a QAAResult."""
        result = qaa_unknown.run()
        assert isinstance(result, QAAResult)

    def test_verified_results_contains_target(self, qaa_unknown):
        """Verified results contain the target value 3."""
        result = qaa_unknown.run()
        assert 3 in result.verified_results

    def test_verified_results_only_valid(self, qaa_unknown):
        """All verified results pass the verify function."""
        result = qaa_unknown.run()
        for r in result.verified_results:
            assert r == 3

    def test_iterations_is_positive(self, qaa_unknown):
        """At least 1 iteration was performed."""
        result = qaa_unknown.run()
        assert result.iterations >= 1

    def test_execution_result_stored(self, qaa_unknown):
        """ExecutionResult is stored in the result."""
        result = qaa_unknown.run()
        assert isinstance(result.execution_result, ExecutionResult)

    def test_finds_target_with_high_shots(self, grover_2q, verify_is_3):
        """With high shots, unknown probability case finds target on first iteration."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=None,
            base=1.5,
            seed=42,
        )
        exec_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 8192},
            skip_transpilation=True,
        )
        executor = AerExecutor(config=exec_config)
        qaa = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)
        result = qaa.run()
        assert 3 in result.verified_results


# === Tests: run with IterativeExecutor ===


class TestRunWithIterativeExecutor:

    def test_known_prob_finds_target_with_single_shot(
        self, iterative_executor, qaa_config_known
    ):
        """QAA with IterativeExecutor and shots=1 finds the target."""
        qaa = QuantumAmplitudeAmplification(
            executor=iterative_executor,
            qaa_config=qaa_config_known,
        )
        result = qaa.run()
        assert 3 in result.verified_results

    def test_unknown_prob_finds_target_with_single_shot(
        self, iterative_executor, qaa_config_unknown
    ):
        """QAA with unknown probability and IterativeExecutor finds the target."""
        qaa = QuantumAmplitudeAmplification(
            executor=iterative_executor,
            qaa_config=qaa_config_unknown,
        )
        result = qaa.run()
        assert 3 in result.verified_results

    def test_iterative_executor_call_count_increases(self, qaa_config_known):
        """IterativeExecutor call_count increases after QAA run."""
        config = ExecutionConfig(run_options={"shots": 1})
        executor = IterativeExecutor(AerExecutor(config), base_seed=50)
        qaa = QuantumAmplitudeAmplification(
            executor=executor,
            qaa_config=qaa_config_known,
        )
        qaa.run()
        assert executor.call_count > 0


# === Tests: Reproducibility ===


class TestReproducibility:

    def test_same_seed_same_result_known_prob(self, grover_2q, verify_is_3):
        """Same seed produces identical results for known probability."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=0.25,
            seed=42,
        )
        exec_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 1024},
            transpiler_options={"seed_transpiler": 42},
            skip_transpilation=True,
        )
        executor = AerExecutor(config=exec_config)

        qaa1 = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)
        qaa2 = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)

        result1 = qaa1.run()
        result2 = qaa2.run()

        assert result1.verified_results == result2.verified_results
        assert result1.iterations == result2.iterations

    def test_same_seed_same_result_unknown_prob(self, grover_2q, verify_is_3):
        """Same seed produces identical results for unknown probability."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=None,
            base=1.5,
            seed=42,
        )
        exec_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 1024},
            transpiler_options={"seed_transpiler": 42},
            skip_transpilation=True,
        )
        executor = AerExecutor(config=exec_config)

        qaa1 = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)
        qaa2 = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)

        result1 = qaa1.run()
        result2 = qaa2.run()

        assert result1.verified_results == result2.verified_results
        assert result1.iterations == result2.iterations

    def test_iterative_executor_reproducible_after_reset(self, qaa_config_known):
        """IterativeExecutor reset allows reproducible re-runs."""
        config = ExecutionConfig(run_options={"shots": 1}, skip_transpilation=True)
        executor = IterativeExecutor(AerExecutor(config), base_seed=77)

        qaa = QuantumAmplitudeAmplification(
            executor=executor,
            qaa_config=qaa_config_known,
        )

        result1 = qaa.run()
        executor.reset()
        result2 = qaa.run()

        assert result1.verified_results == result2.verified_results
        assert result1.iterations == result2.iterations

    def test_known_probability_needs_multiple_attempts_with_suboptimal_iterations(self):
        """Overestimated success_prob gives 0 Grover iterations, forcing repeated measurement."""
        unitary, grover = _build_2qubit_grover_for_target_11()

        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=lambda x: x == 3,
            success_prob=0.9,
            seed=42,
        )
        exec_config = ExecutionConfig(run_options={"shots": 1}, skip_transpilation=True)
        executor = IterativeExecutor(AerExecutor(exec_config), base_seed=0)

        qaa = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)
        result1 = qaa.run()

        assert 3 in result1.verified_results
        assert result1.iterations == 3

        executor.reset()
        result2 = qaa.run()

        assert result1.iterations == result2.iterations
        assert result1.verified_results == result2.verified_results


# === Tests: run does not corrupt global random state ===


class TestRandomStateIsolation:

    def test_run_is_independent_of_external_random_state(self, qaa_known):
        """QAA results depend only on its configured seed, not the external random state."""
        import random

        # Run with one external random state
        random.seed(111)
        result1 = qaa_known.run()

        # Run with a completely different external random state
        random.seed(999)
        result2 = qaa_known.run()

        # QAA uses its own internal seed (42), so results must be identical
        assert result1.verified_results == result2.verified_results
        assert result1.iterations == result2.iterations


# === Tests: Edge cases ===


class TestEdgeCases:

    def test_verify_accepts_all_returns_immediately(self, aer_executor, grover_2q):
        """Verify that accepts everything returns on first measurement."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=lambda x: True,
            success_prob=0.25,
            seed=42,
        )
        qaa = QuantumAmplitudeAmplification(executor=aer_executor, qaa_config=config)
        result = qaa.run()
        assert len(result.verified_results) > 0
        assert result.iterations == 1

    def test_single_shot_known_probability(self, grover_2q, verify_is_3):
        """QAA works with single shot executor (known probability)."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=0.25,
            seed=42,
        )
        exec_config = ExecutionConfig(run_options={"shots": 1}, skip_transpilation=True)
        executor = IterativeExecutor(AerExecutor(exec_config), base_seed=42)
        qaa = QuantumAmplitudeAmplification(executor=executor, qaa_config=config)
        result = qaa.run()
        assert 3 in result.verified_results

    def test_3_qubit_search(self, aer_executor):
        """QAA works on a 3-qubit system searching for |111⟩ (=7)."""
        unitary = QuantumCircuit(3)
        unitary.h([0, 1, 2])

        grover = QuantumCircuit(3)
        # Oracle: mark |111⟩ with CCZ
        grover.h(2)
        grover.ccx(0, 1, 2)
        grover.h(2)
        # Diffusion
        grover.h([0, 1, 2])
        grover.x([0, 1, 2])
        grover.h(2)
        grover.ccx(0, 1, 2)
        grover.h(2)
        grover.x([0, 1, 2])
        grover.h([0, 1, 2])

        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=3,
            verify=lambda x: x == 7,
            success_prob=1 / 8,
            seed=42,
        )
        qaa = QuantumAmplitudeAmplification(executor=aer_executor, qaa_config=config)
        result = qaa.run()
        assert 7 in result.verified_results

    def test_unknown_probability_without_base(
        self, aer_executor, grover_2q, verify_is_3
    ):
        """Unknown probability with base=None still finds target."""
        unitary, grover = grover_2q
        config = QAAConfig(
            unitary=unitary,
            grover_circuit=grover,
            qubit_count=2,
            verify=verify_is_3,
            success_prob=None,
            base=None,
            seed=42,
        )
        qaa = QuantumAmplitudeAmplification(executor=aer_executor, qaa_config=config)
        result = qaa.run()
        assert 3 in result.verified_results

    def test_build_circuit_with_large_repetitions(self, qaa_known):
        """build_circuit handles large repetition counts without error."""
        circuit = qaa_known.build_circuit(repetitions=10)
        assert circuit.num_qubits == 2
        assert circuit.num_clbits == 2
        assert circuit.size() > qaa_known.build_circuit(repetitions=1).size()
