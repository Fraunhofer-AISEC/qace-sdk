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
import pytest

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from beartype.roar import BeartypeCallHintParamViolation

from qace.algorithm import (
    CorrelationExtraction,
    CorrelationExtractionResult,
    AlgorithmResult,
)
from qace.execution import (
    AerExecutor,
    ExecutionConfig,
    ExecutionResult,
    IBMRuntimeExecutor,
    IBMRuntimeExecutionConfig,
)
from qace.vbf import VectorialBooleanFunctionFull

# === Fixtures ===


@pytest.fixture
def identity_vbf():
    """Identity function on 2 bits: f(x) = x."""
    return VectorialBooleanFunctionFull(lambda x: x, 2, 2)


@pytest.fixture
def constant_zero_vbf():
    """Constant zero function on 2 bits: f(x) = 0."""
    return VectorialBooleanFunctionFull(lambda x: 0, 2, 2)


@pytest.fixture
def xor_vbf():
    """XOR function: f(x) = x0 XOR x1 (2 input bits, 1 output bit)."""
    return VectorialBooleanFunctionFull(lambda x: ((x >> 1) & 1) ^ (x & 1), 2, 1)


@pytest.fixture
def aer_executor():
    """Seeded AerExecutor for reproducibility."""
    config = ExecutionConfig(
        backend_options={"seed_simulator": 42},
        run_options={"shots": 1024},
        transpiler_options={"seed_transpiler": 42},
    )
    return AerExecutor(config=config)


@pytest.fixture
def high_shots_executor():
    """Executor with high shot count for statistical tests."""
    config = ExecutionConfig(
        backend_options={"seed_simulator": 42},
        run_options={"shots": 8192},
        transpiler_options={"seed_transpiler": 42},
    )
    return AerExecutor(config=config)


@pytest.fixture
def cea_identity(aer_executor, identity_vbf):
    """CEA with identity VBF."""
    return CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)


@pytest.fixture
def cea_constant(aer_executor, constant_zero_vbf):
    """CEA with constant zero VBF."""
    return CorrelationExtraction(executor=aer_executor, vbf=constant_zero_vbf)


# === Tests: Initialization ===


class TestCorrelationExtractionInit:

    def test_executor_stored(self, aer_executor, identity_vbf):
        """Executor is stored on the instance."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        assert cea._executor is aer_executor

    def test_vbf_stored(self, aer_executor, identity_vbf):
        """VBF is stored on the instance."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        assert cea.vbf is identity_vbf

    def test_vbf_m_accessible(self, aer_executor, identity_vbf):
        """VBF domain size m is accessible."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        assert cea.vbf.m == 2

    def test_vbf_n_accessible(self, aer_executor, identity_vbf):
        """VBF image size n is accessible."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        assert cea.vbf.n == 2

    def test_asymmetric_dimensions(self, aer_executor, xor_vbf):
        """CEA works with asymmetric m != n."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=xor_vbf)
        assert cea.vbf.m == 2
        assert cea.vbf.n == 1


# === Tests: Type checking on init (beartype instruments _cea.py) ===


class TestCorrelationExtractionTypeChecking:

    def test_none_executor_raises(self, identity_vbf):
        """None executor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            CorrelationExtraction(executor=None, vbf=identity_vbf)

    def test_wrong_type_executor_raises(self, identity_vbf):
        """Non-CircuitExecutor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            CorrelationExtraction(executor="not_an_executor", vbf=identity_vbf)

    def test_none_vbf_raises(self, aer_executor):
        """None VBF raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            CorrelationExtraction(executor=aer_executor, vbf=None)

    def test_wrong_type_vbf_raises(self, aer_executor):
        """Non-VBF raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            CorrelationExtraction(executor=aer_executor, vbf="not_a_vbf")


# === Tests: build_circuit ===


class TestCorrelationExtractionBuildCircuit:

    def test_correct_qubit_count(self, cea_identity):
        """Circuit has m + n qubits."""
        circuit = cea_identity.build_circuit()
        assert circuit.num_qubits == 4

    def test_correct_clbit_count(self, cea_identity):
        """Circuit has m + n classical bits."""
        circuit = cea_identity.build_circuit()
        assert circuit.num_clbits == 4

    def test_has_measurements(self, cea_identity):
        """Circuit contains measurement operations."""
        circuit = cea_identity.build_circuit()
        op_names = [inst.operation.name for inst in circuit.data]
        assert "measure" in op_names

    def test_measurement_count_equals_num_qubits(self, cea_identity):
        """Circuit has exactly m + n measurements."""
        circuit = cea_identity.build_circuit()
        measure_count = sum(
            1 for inst in circuit.data if inst.operation.name == "measure"
        )
        assert measure_count == cea_identity.vbf.m + cea_identity.vbf.n

    def test_asymmetric_qubit_count(self, aer_executor, xor_vbf):
        """Circuit handles m != n correctly (m=2, n=1 -> 3 qubits)."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=xor_vbf)
        circuit = cea.build_circuit()
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3

    def test_idempotent(self, cea_identity):
        """Multiple calls return structurally equivalent circuits."""
        circuit1 = cea_identity.build_circuit()
        circuit2 = cea_identity.build_circuit()
        assert circuit1.num_qubits == circuit2.num_qubits
        assert circuit1.num_clbits == circuit2.num_clbits
        assert circuit1.depth() == circuit2.depth()
        assert circuit1.size() == circuit2.size()


# === Tests: _build_cea_unitary ===


class TestCorrelationExtractionUnitary:

    def test_no_measurements(self, cea_identity):
        """CEA unitary has no measurement gates."""
        unitary = cea_identity.build_cea_unitary()
        op_names = [inst.operation.name for inst in unitary.data]
        assert "measure" not in op_names

    def test_correct_qubit_count(self, cea_identity):
        """CEA unitary has m + n qubits."""
        unitary = cea_identity.build_cea_unitary()
        assert unitary.num_qubits == 4

    def test_no_classical_bits(self, cea_identity):
        """CEA unitary has no classical bits."""
        unitary = cea_identity.build_cea_unitary()
        assert unitary.num_clbits == 0

    def test_contains_hadamard_gates(self, cea_identity):
        """Unitary contains Hadamard gates."""
        unitary = cea_identity.build_cea_unitary()
        op_names = [inst.operation.name for inst in unitary.data]
        assert "h" in op_names

    def test_hadamard_count_at_least_m_plus_mn(self, aer_executor):
        """Unitary has at least m + (m+n) Hadamard gates."""
        vbf = VectorialBooleanFunctionFull(lambda x: 0, 2, 2)
        cea = CorrelationExtraction(executor=aer_executor, vbf=vbf)
        unitary = cea.build_cea_unitary()
        h_count = sum(1 for inst in unitary.data if inst.operation.name == "h")
        assert h_count == 2 * vbf.m + vbf.n


# === Tests: _counts_to_mask_pairs ===


class TestCountsToMaskPairs:

    def test_empty_counts(self, cea_identity):
        """Empty counts returns empty dict."""
        result = cea_identity._counts_to_mask_pairs({})
        assert result == {}

    def test_single_binary_bitstring(self, cea_identity):
        """Single binary bitstring correctly parsed (m=2, n=2)."""
        # "1101" -> value=13 -> alpha=(13>>2)&3=3, beta=13&3=1
        counts = {"1101": 5}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(3, 1): 5}

    def test_single_hex_bitstring(self, cea_identity):
        """Single hex bitstring correctly parsed."""
        # "0xd" -> value=13 -> alpha=3, beta=1
        counts = {"0xd": 10}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(3, 1): 10}

    def test_multiple_bitstrings(self, cea_identity):
        """Multiple bitstrings produce correct mask pairs."""
        counts = {"0000": 100, "1111": 200}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(0, 0): 100, (3, 3): 200}

    def test_all_zeros(self, cea_identity):
        """All-zero bitstring gives (0, 0)."""
        counts = {"0000": 50}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(0, 0): 50}

    def test_all_ones(self, cea_identity):
        """All-one bitstring gives (3, 3) for m=2, n=2."""
        counts = {"1111": 77}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(3, 3): 77}

    def test_asymmetric_split(self, aer_executor, xor_vbf):
        """Asymmetric m=2, n=1: correct alpha/beta split."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=xor_vbf)
        # "101" -> value=5 -> alpha=(5>>1)&3=2, beta=5&1=1
        counts = {"101": 7}
        result = cea._counts_to_mask_pairs(counts)
        assert result == {(2, 1): 7}

    def test_preserves_count_values(self, cea_identity):
        """Count values are preserved unchanged."""
        counts = {"0001": 1, "0010": 2, "0100": 3, "1000": 4}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert sum(result.values()) == 10

    def test_all_16_pairs_for_2bit(self, cea_identity):
        """All 16 possible (alpha, beta) pairs for m=2, n=2."""
        counts = {format(i, "04b"): 1 for i in range(16)}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert len(result) == 16
        for a in range(4):
            for b in range(4):
                assert (a, b) in result

    def test_hex_prefix_detection(self, cea_identity):
        """Hex strings with 0x prefix are parsed as hex."""
        counts = {"0x0": 10, "0xf": 20}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert (0, 0) in result
        assert (3, 3) in result

    def test_binary_without_prefix(self, cea_identity):
        """Binary strings without prefix are parsed as binary."""
        # "0101" -> value=5 -> alpha=(5>>2)&3=1, beta=5&3=1
        counts = {"0101": 42}
        result = cea_identity._counts_to_mask_pairs(counts)
        assert result == {(1, 1): 42}


# === Tests: run ===


class TestCorrelationExtractionRun:

    def test_returns_correlation_extraction_result(self, cea_identity):
        """run() returns a CorrelationExtractionResult."""
        result = cea_identity.run()
        assert isinstance(result, CorrelationExtractionResult)

    def test_result_is_algorithm_result(self, cea_identity):
        """CorrelationExtractionResult is an AlgorithmResult."""
        result = cea_identity.run()
        assert isinstance(result, AlgorithmResult)

    def test_mask_pairs_is_dict(self, cea_identity):
        """Result contains mask_pairs as a dict."""
        result = cea_identity.run()
        assert isinstance(result.mask_pairs, dict)

    def test_mask_pairs_not_empty(self, cea_identity):
        """Mask pairs are not empty after execution."""
        result = cea_identity.run()
        assert len(result.mask_pairs) > 0

    def test_mask_pairs_keys_are_int_tuples(self, cea_identity):
        """Mask pair keys are (int, int) tuples."""
        result = cea_identity.run()
        for key in result.mask_pairs.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], int)

    def test_mask_pairs_values_are_positive_ints(self, cea_identity):
        """Mask pair values are positive integer counts."""
        result = cea_identity.run()
        for value in result.mask_pairs.values():
            assert isinstance(value, int)
            assert value > 0

    def test_mask_pairs_sum_equals_shots(self, cea_identity):
        """Sum of mask pair counts equals total shots."""
        result = cea_identity.run()
        total = sum(result.mask_pairs.values())
        assert total == 1024

    def test_execution_result_stored(self, cea_identity):
        """ExecutionResult is stored in the result."""
        result = cea_identity.run()
        assert isinstance(result.execution_result, ExecutionResult)

    def test_alpha_within_range(self, cea_identity):
        """All alpha values are in [0, 2^m - 1]."""
        result = cea_identity.run()
        for alpha, _ in result.mask_pairs.keys():
            assert 0 <= alpha < (1 << 2)

    def test_beta_within_range(self, cea_identity):
        """All beta values are in [0, 2^n - 1]."""
        result = cea_identity.run()
        for _, beta in result.mask_pairs.keys():
            assert 0 <= beta < (1 << 2)


# === Tests: Reproducibility ===


class TestCorrelationExtractionReproducibility:

    def test_seeded_execution_reproducible(self, aer_executor, identity_vbf):
        """Same seed produces identical mask pairs."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        result1 = cea.run()
        result2 = cea.run()
        assert result1.mask_pairs == result2.mask_pairs

    def test_different_seeds_may_differ(self, identity_vbf):
        """Different seeds produce different count distributions."""
        config1 = ExecutionConfig(
            backend_options={"seed_simulator": 1},
            run_options={"shots": 4096},
        )
        config2 = ExecutionConfig(
            backend_options={"seed_simulator": 9999},
            run_options={"shots": 4096},
        )
        cea1 = CorrelationExtraction(executor=AerExecutor(config1), vbf=identity_vbf)
        cea2 = CorrelationExtraction(executor=AerExecutor(config2), vbf=identity_vbf)

        result1 = cea1.run()
        result2 = cea2.run()
        assert result1.mask_pairs != result2.mask_pairs


# === Tests: Mathematical properties of CEA ===


class TestCorrelationExtractionMathematical:

    def test_constant_zero_alpha_always_zero(
        self, high_shots_executor, constant_zero_vbf
    ):
        """For f(x)=0, alpha is always 0.

        H^m undoes the initial H^m since f doesn't entangle the registers.
        """
        cea = CorrelationExtraction(executor=high_shots_executor, vbf=constant_zero_vbf)
        result = cea.run()

        for alpha, beta in result.mask_pairs.keys():
            assert alpha == 0

    def test_constant_zero_all_beta(self, high_shots_executor, constant_zero_vbf):
        """For f(x)=0, all 2^n beta values appear."""
        cea = CorrelationExtraction(executor=high_shots_executor, vbf=constant_zero_vbf)
        result = cea.run()

        n = constant_zero_vbf.n
        betas = {beta for (_, beta) in result.mask_pairs.keys()}
        assert len(betas) == (1 << n)

    def test_constant_zero_beta_approximately_uniform(
        self, high_shots_executor, constant_zero_vbf
    ):
        """For f(x)=0, beta counts are approximately uniform."""
        cea = CorrelationExtraction(executor=high_shots_executor, vbf=constant_zero_vbf)
        result = cea.run()

        n = constant_zero_vbf.n
        expected_per_beta = 8192 / (1 << n)  # 8192 / 4 = 2048
        for count in result.mask_pairs.values():
            assert abs(count - expected_per_beta) / expected_per_beta < 0.2


# === Tests: Different VBFs produce different results ===


class TestCorrelationExtractionDifferentVBFs:

    def test_identity_vs_constant_different(
        self, aer_executor, identity_vbf, constant_zero_vbf
    ):
        """Different VBFs produce different mask pair distributions."""
        cea_id = CorrelationExtraction(executor=aer_executor, vbf=identity_vbf)
        cea_const = CorrelationExtraction(executor=aer_executor, vbf=constant_zero_vbf)

        result_id = cea_id.run()
        result_const = cea_const.run()

        assert result_id.mask_pairs != result_const.mask_pairs

    def test_xor_function_valid_ranges(self, aer_executor, xor_vbf):
        """CEA on XOR (m=2, n=1) produces valid ranges."""
        cea = CorrelationExtraction(executor=aer_executor, vbf=xor_vbf)
        result = cea.run()

        assert len(result.mask_pairs) > 0
        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 4
            assert 0 <= beta < 2

    def test_larger_function_3bit(self, aer_executor):
        """CEA works on a 3-bit to 3-bit function."""
        vbf = VectorialBooleanFunctionFull(lambda x: x ^ 0b101, 3, 3)
        cea = CorrelationExtraction(executor=aer_executor, vbf=vbf)
        result = cea.run()

        assert len(result.mask_pairs) > 0
        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 8
            assert 0 <= beta < 8


# === Tests: CorrelationExtractionResult dataclass ===


class TestCorrelationExtractionResult:

    def test_default_mask_pairs_empty(self):
        """Default mask_pairs is an empty dict."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        result = CorrelationExtractionResult(
            execution_result=exec_result,
            mask_pairs={},
        )
        assert result.mask_pairs == {}

    def test_mask_pairs_stored(self):
        """Custom mask_pairs are stored correctly."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        pairs = {(0, 0): 100, (1, 1): 200}
        result = CorrelationExtractionResult(
            execution_result=exec_result,
            mask_pairs=pairs,
        )
        assert result.mask_pairs == pairs

    def test_inherits_algorithm_result_fields(self):
        """CorrelationExtractionResult has all AlgorithmResult fields."""
        exec_result = ExecutionResult(counts={"00": 50}, result=None, metadata={})
        result = CorrelationExtractionResult(
            execution_result=exec_result,
            metadata={"m": 2, "n": 2},
            mask_pairs={(0, 0): 50},
        )
        assert result.execution_result is exec_result
        assert result.metadata == {"m": 2, "n": 2}
        assert result.mask_pairs == {(0, 0): 50}

    def test_mask_pairs_not_shared_between_instances(self):
        """Each instance has its own mask_pairs dict."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        r1 = CorrelationExtractionResult(execution_result=exec_result)
        r2 = CorrelationExtractionResult(execution_result=exec_result)
        r1.mask_pairs[(1, 2)] = 5
        assert (1, 2) not in r2.mask_pairs


# === Tests: Edge cases ===


class TestCorrelationExtractionEdgeCases:

    def test_single_shot(self, identity_vbf):
        """CEA works with a single shot."""
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 1},
        )
        executor = AerExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=identity_vbf)
        result = cea.run()

        assert sum(result.mask_pairs.values()) == 1
        assert len(result.mask_pairs) == 1

    def test_minimal_1bit_function(self):
        """CEA works on minimal 1-bit to 1-bit function."""
        vbf = VectorialBooleanFunctionFull(lambda x: x, 1, 1)
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 100},
        )
        executor = AerExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=vbf)
        result = cea.run()

        assert len(result.mask_pairs) > 0
        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 2
            assert 0 <= beta < 2
        assert sum(result.mask_pairs.values()) == 100

    def test_high_shot_count(self, identity_vbf):
        """CEA handles high shot counts."""
        s = 50000
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": s},
        )
        executor = AerExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=identity_vbf)
        result = cea.run()

        assert sum(result.mask_pairs.values()) == s

    def test_4bit_function(self):
        """CEA works on a 4-bit to 4-bit function."""
        vbf = VectorialBooleanFunctionFull(lambda x: (x * 3) & 0xF, 4, 4)
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 2048},
        )
        executor = AerExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=vbf)
        result = cea.run()

        assert len(result.mask_pairs) > 0
        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 16
            assert 0 <= beta < 16
        assert sum(result.mask_pairs.values()) == 2048


# === Fixtures for IBM Runtime ===


@pytest.fixture
def fake_backend():
    """Small 5-qubit fake backend."""
    return FakeManilaV2()


@pytest.fixture
def runtime_executor(fake_backend):
    """Seeded IBMRuntimeExecutor for reproducibility."""
    config = IBMRuntimeExecutionConfig(
        backend_options={"backend": fake_backend},
        run_options={"shots": 1024},
        sampler_options={"simulator": {"seed_simulator": 42}},
        transpiler_options={"seed_transpiler": 42},
    )
    return IBMRuntimeExecutor(config=config)


@pytest.fixture
def runtime_high_shots_executor(fake_backend):
    """IBMRuntimeExecutor with high shot count."""
    config = IBMRuntimeExecutionConfig(
        backend_options={"backend": fake_backend},
        run_options={"shots": 8192},
        sampler_options={"simulator": {"seed_simulator": 42}},
        transpiler_options={"seed_transpiler": 42},
    )
    return IBMRuntimeExecutor(config=config)


# === Tests: CEA with IBM Runtime Executor ===


class TestCorrelationExtractionIBMRuntime:

    def test_run_returns_result(self, runtime_executor, identity_vbf):
        """CEA produces a CorrelationExtractionResult with IBMRuntimeExecutor."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result = cea.run()
        assert isinstance(result, CorrelationExtractionResult)

    def test_mask_pairs_not_empty(self, runtime_executor, identity_vbf):
        """Mask pairs are not empty."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result = cea.run()
        assert len(result.mask_pairs) > 0

    def test_mask_pairs_sum_equals_shots(self, runtime_executor, identity_vbf):
        """Sum of mask pair counts equals total shots."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result = cea.run()
        assert sum(result.mask_pairs.values()) == 1024

    def test_alpha_beta_within_range(self, runtime_executor, identity_vbf):
        """All (alpha, beta) are within valid range."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result = cea.run()

        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 4
            assert 0 <= beta < 4

    def test_reproducible_with_seed(self, runtime_executor, identity_vbf):
        """Same seed produces identical results."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result1 = cea.run()
        result2 = cea.run()
        assert result1.mask_pairs == result2.mask_pairs

    def test_asymmetric_function(self, runtime_executor, xor_vbf):
        """CEA with m != n works on IBM Runtime."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=xor_vbf)
        result = cea.run()

        assert len(result.mask_pairs) > 0
        for alpha, beta in result.mask_pairs.keys():
            assert 0 <= alpha < 4
            assert 0 <= beta < 2

    def test_constant_zero_alpha_always_zero(
        self, runtime_high_shots_executor, constant_zero_vbf
    ):
        """For f(x)=0, alpha is always 0 (even with noisy backend)."""
        cea = CorrelationExtraction(
            executor=runtime_high_shots_executor, vbf=constant_zero_vbf
        )
        result = cea.run()

        # With noise, non-zero alpha may appear, but (0, *) should dominate
        total = sum(result.mask_pairs.values())
        alpha_zero_count = sum(
            count for (alpha, _), count in result.mask_pairs.items() if alpha == 0
        )
        assert alpha_zero_count / total > 0.8

    def test_noise_affects_distribution(self, identity_vbf):
        """Noisy IBM Runtime produces different results than ideal Aer."""
        # Ideal (Aer, no noise)
        ideal_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 8192},
            transpiler_options={"seed_transpiler": 42},
        )
        cea_ideal = CorrelationExtraction(
            executor=AerExecutor(ideal_config), vbf=identity_vbf
        )

        # Noisy (IBM Runtime, fake backend)
        noisy_config = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeManilaV2()},
            run_options={"shots": 8192},
            sampler_options={"simulator": {"seed_simulator": 42}},
            transpiler_options={"seed_transpiler": 42},
        )
        cea_noisy = CorrelationExtraction(
            executor=IBMRuntimeExecutor(noisy_config), vbf=identity_vbf
        )

        result_ideal = cea_ideal.run()
        result_noisy = cea_noisy.run()

        assert result_ideal.mask_pairs != result_noisy.mask_pairs

    def test_mask_pairs_consistent_with_execution_counts(
        self, runtime_executor, identity_vbf
    ):
        """Mask pairs are a correct conversion of execution_result.counts."""
        cea = CorrelationExtraction(executor=runtime_executor, vbf=identity_vbf)
        result = cea.run()

        # Manually convert counts using the same logic
        m = identity_vbf.m
        n = identity_vbf.n
        expected_mask_pairs: dict[tuple[int, int], int] = {}

        for bitstring, count in result.execution_result.counts.items():
            value = (
                int(bitstring, 16) if bitstring.startswith("0x") else int(bitstring, 2)
            )
            alpha = (value >> n) & ((1 << m) - 1)
            beta = value & ((1 << n) - 1)
            expected_mask_pairs[(alpha, beta)] = count

        assert result.mask_pairs == expected_mask_pairs

    def test_different_backends_different_results(self, identity_vbf):
        """Different fake backends produce different distributions."""
        from qiskit_ibm_runtime.fake_provider import FakeNairobiV2

        config_manila = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeManilaV2()},
            run_options={"shots": 8192},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )
        config_nairobi = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeNairobiV2()},
            run_options={"shots": 8192},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )

        cea_manila = CorrelationExtraction(
            executor=IBMRuntimeExecutor(config_manila), vbf=identity_vbf
        )
        cea_nairobi = CorrelationExtraction(
            executor=IBMRuntimeExecutor(config_nairobi), vbf=identity_vbf
        )

        result_manila = cea_manila.run()
        result_nairobi = cea_nairobi.run()

        assert result_manila.mask_pairs != result_nairobi.mask_pairs

    def test_mask_pairs_match_known_test_vectors(self, identity_vbf):
        """Mask pairs match pre-computed expected values for a seeded run."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeManilaV2()},
            run_options={"shots": 8192},
            sampler_options={"simulator": {"seed_simulator": 42}},
            transpiler_options={"seed_transpiler": 24},
        )
        executor = IBMRuntimeExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=identity_vbf)
        result = cea.run()

        expected_counts = {
            "0000": 1654,
            "0001": 308,
            "0010": 179,
            "0011": 69,
            "0100": 255,
            "0101": 1615,
            "0110": 68,
            "0111": 165,
            "1000": 135,
            "1001": 69,
            "1010": 1541,
            "1011": 329,
            "1100": 59,
            "1101": 143,
            "1110": 228,
            "1111": 1375,
        }

        expected_mask_pairs = {
            (0, 0): 1654,
            (0, 1): 308,
            (0, 2): 179,
            (0, 3): 69,
            (1, 0): 255,
            (1, 1): 1615,
            (1, 2): 68,
            (1, 3): 165,
            (2, 0): 135,
            (2, 1): 69,
            (2, 2): 1541,
            (2, 3): 329,
            (3, 0): 59,
            (3, 1): 143,
            (3, 2): 228,
            (3, 3): 1375,
        }

        assert expected_mask_pairs == cea._counts_to_mask_pairs(expected_counts)
        assert result.execution_result.counts == expected_counts
        assert result.mask_pairs == expected_mask_pairs


# === Tests: CEA with Aer Executor test vector ===


class TestCorrelationExtractionAerTestVectors:

    def test_mask_pairs_match_known_test_vectors(self, identity_vbf):
        """Mask pairs match pre-computed expected values for a seeded Aer run."""
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 1024},
            transpiler_options={"seed_transpiler": 42},
        )
        executor = AerExecutor(config=config)
        cea = CorrelationExtraction(executor=executor, vbf=identity_vbf)
        result = cea.run()

        expected_counts = {"1111": 254, "0101": 260, "1010": 264, "0000": 246}

        expected_mask_pairs = {(0, 0): 246, (1, 1): 260, (2, 2): 264, (3, 3): 254}

        assert result.execution_result.counts == expected_counts
        assert result.mask_pairs == expected_mask_pairs
