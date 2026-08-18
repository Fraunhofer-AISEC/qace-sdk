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
"""Tests for the Preimage search module (_preimage.py)."""

import pytest
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from beartype.roar import BeartypeCallHintParamViolation

from qace.algorithm import (
    QuantumAlgorithm,
    PreimageResult,
    Preimage,
    PreimageConfig,
    build_phase_flip_x_circuit,
)
from qace.execution import (
    AerExecutor,
    ExecutionConfig,
    ExecutionResult,
    IterativeExecutor,
)
from qace.vbf import VectorialBooleanFunctionFull

# === Fixtures ===


@pytest.fixture
def identity_2bit_vbf():
    """Identity function on 2 bits: f(x) = x."""
    return VectorialBooleanFunctionFull(lambda x: x, 2, 2)


@pytest.fixture
def constant_zero_2bit_vbf():
    """Constant zero function: f(x) = 0, m=2, n=2."""
    return VectorialBooleanFunctionFull(lambda x: 0, 2, 2)


@pytest.fixture
def xor_one_2bit_vbf():
    """XOR with 1 function: f(x) = x ^ 1, m=2, n=2."""
    return VectorialBooleanFunctionFull(lambda x: x ^ 1, 2, 2)


@pytest.fixture
def aer_executor():
    """Seeded AerExecutor for reproducibility."""
    config = ExecutionConfig(
        backend_options={"seed_simulator": 42},
        run_options={"shots": 1024},
        skip_transpilation=True,
    )
    return AerExecutor(config=config)


@pytest.fixture
def iterative_executor():
    """IterativeExecutor with single shot for deterministic behavior."""
    config = ExecutionConfig(
        run_options={"shots": 1}, transpiler_options={"seed_transpiler": 42}
    )
    return IterativeExecutor(AerExecutor(config), base_seed=100)


@pytest.fixture
def preimage_config_identity_target_3(identity_2bit_vbf):
    """PreimageConfig for identity VBF targeting image=3."""
    return PreimageConfig(
        vbf=identity_2bit_vbf,
        vbf_image=3,
        success_prob=0.25,
        seed=42,
    )


@pytest.fixture
def preimage_config_no_optionals(identity_2bit_vbf):
    """PreimageConfig with only required fields."""
    return PreimageConfig(
        vbf=identity_2bit_vbf,
        vbf_image=0,
    )


@pytest.fixture
def preimage_instance(aer_executor, preimage_config_identity_target_3):
    """Preimage instance with identity VBF targeting image=3."""
    return Preimage(
        executor=aer_executor,
        preimage_config=preimage_config_identity_target_3,
    )


# === Tests: build_phase_flip_x_circuit structure ===


class TestBuildPhaseFlipXCircuitStructure:

    def test_returns_quantum_circuit(self):
        """build_phase_flip_x_circuit returns a QuantumCircuit."""
        circuit = build_phase_flip_x_circuit(2, 0)
        assert isinstance(circuit, QuantumCircuit)

    def test_qubit_count_is_n_plus_one(self):
        """Circuit has n + 1 qubits (n control + 1 auxiliary)."""
        circuit = build_phase_flip_x_circuit(3, 0)
        assert circuit.num_qubits == 4

    def test_different_x_values_produce_different_circuits(self):
        """Different target states produce structurally different circuits."""
        c0 = build_phase_flip_x_circuit(3, 0)
        c5 = build_phase_flip_x_circuit(3, 5)
        # The circuits differ in the ctrl_state of the MCX gate
        assert c0 != c5

    def test_idempotent_same_parameters(self):
        """Same parameters produce structurally identical circuits."""
        c1 = build_phase_flip_x_circuit(3, 5)
        c2 = build_phase_flip_x_circuit(3, 5)
        assert c1 == c2


# === Tests: build_phase_flip_x_circuit simulation ===


class TestBuildPhaseFlipXCircuitBehavior:

    def test_flips_phase_of_zero_state(self):
        """Phase flip for x=0 negates amplitude of |0...0> in superposition."""
        n = 2
        x = 0
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(range(n))
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2**n)

        # Target state |00>|0> = Qiskit index 0
        assert sv[0] == pytest.approx(-expected_amp, abs=1e-10)
        # Other states unchanged
        assert sv[1] == pytest.approx(expected_amp, abs=1e-10)
        assert sv[2] == pytest.approx(expected_amp, abs=1e-10)
        assert sv[3] == pytest.approx(expected_amp, abs=1e-10)

    def test_flips_phase_of_all_ones_state(self):
        """Phase flip for x=2^n-1 negates amplitude of |11> in superposition."""
        n = 2
        x = 3
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(range(n))
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2**n)

        # |11>|0> in project convention: q0=1, q1=1, aux=0
        # Qiskit index: 1*1 + 1*2 + 0*4 = 3
        assert sv[3] == pytest.approx(-expected_amp, abs=1e-10)
        # Other states unchanged
        assert sv[0] == pytest.approx(expected_amp, abs=1e-10)
        assert sv[1] == pytest.approx(expected_amp, abs=1e-10)
        assert sv[2] == pytest.approx(expected_amp, abs=1e-10)

    def test_leaves_non_target_states_unchanged(self):
        """Phase flip only affects the target state, not others."""
        n = 3
        x = 0
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(range(n))
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2**n)

        # All non-target states should have positive amplitude
        for idx in range(1, 2**n):
            assert sv[idx] == pytest.approx(expected_amp, abs=1e-10)

    def test_aux_qubit_returns_to_zero(self):
        """Auxiliary qubit returns to |0> after the phase flip."""
        n = 2
        x = 1
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(range(n))
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)

        # All amplitudes with aux=1 (indices >= 2^n) should be zero
        for idx in range(2**n, 2 ** (n + 1)):
            assert sv[idx] == pytest.approx(0.0, abs=1e-10)

    def test_single_qubit_n_equals_1_flips_zero(self):
        """For n=1, x=0: flips phase of |0>."""
        n = 1
        x = 0
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(0)
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2)

        # |0>|0> = Qiskit index 0
        assert sv[0] == pytest.approx(-expected_amp, abs=1e-10)
        # |1>|0> = Qiskit index 1
        assert sv[1] == pytest.approx(expected_amp, abs=1e-10)

    def test_single_qubit_n_equals_1_flips_one(self):
        """For n=1, x=1: flips phase of |1>."""
        n = 1
        x = 1
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(0)
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2)

        # |1> in project convention: q0=1, aux=0
        # Qiskit index: 1*1 + 0*2 = 1
        assert sv[1] == pytest.approx(-expected_amp, abs=1e-10)
        # |0> unchanged
        assert sv[0] == pytest.approx(expected_amp, abs=1e-10)

    def test_applying_twice_gives_identity(self):
        """Applying the same phase flip twice restores original state."""
        n = 2
        x = 2
        pfc = build_phase_flip_x_circuit(n, x)

        test_circuit = QuantumCircuit(n + 1)
        test_circuit.h(range(n))
        test_circuit.compose(pfc, inplace=True)
        test_circuit.compose(pfc, inplace=True)

        sv = Statevector(test_circuit)
        expected_amp = 1.0 / np.sqrt(2**n)

        # All amplitudes should be restored to positive
        for idx in range(2**n):
            assert sv[idx] == pytest.approx(expected_amp, abs=1e-10)


# === Tests: PreimageConfig ===


class TestPreimageConfig:

    def test_vbf_stored(self, identity_2bit_vbf):
        """VBF is stored correctly."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=0)
        assert config.vbf is identity_2bit_vbf

    def test_vbf_image_stored(self, identity_2bit_vbf):
        """Target image is stored correctly."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=3)
        assert config.vbf_image == 3

    def test_default_success_prob_is_none(self, identity_2bit_vbf):
        """Default success_prob is None."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=0)
        assert config.success_prob is None

    def test_default_seed_is_none(self, identity_2bit_vbf):
        """Default seed is None."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=0)
        assert config.seed is None

    def test_default_base_is_none(self, identity_2bit_vbf):
        """Default base is None."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=0)
        assert config.base is None

    def test_custom_success_prob_stored(self, identity_2bit_vbf):
        """Custom success_prob is stored correctly."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=1, success_prob=0.5)
        assert config.success_prob == 0.5

    def test_custom_seed_stored(self, identity_2bit_vbf):
        """Custom seed is stored correctly."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=1, seed=99)
        assert config.seed == 99

    def test_custom_base_stored(self, identity_2bit_vbf):
        """Custom base is stored correctly."""
        config = PreimageConfig(vbf=identity_2bit_vbf, vbf_image=1, base=1.5)
        assert config.base == 1.5

    def test_all_fields_stored(self, identity_2bit_vbf):
        """All fields are stored when provided."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=2,
            success_prob=0.25,
            seed=42,
            base=1.7,
        )
        assert config.vbf is identity_2bit_vbf
        assert config.vbf_image == 2
        assert config.success_prob == 0.25
        assert config.seed == 42
        assert config.base == 1.7


# === Tests: Preimage initialization ===


class TestPreimageInit:

    def test_executor_stored(self, aer_executor, preimage_config_identity_target_3):
        """Executor is stored on the instance."""
        preimage = Preimage(
            executor=aer_executor,
            preimage_config=preimage_config_identity_target_3,
        )
        assert preimage._executor is aer_executor

    def test_config_stored(self, aer_executor, preimage_config_identity_target_3):
        """PreimageConfig is stored on the instance."""
        preimage = Preimage(
            executor=aer_executor,
            preimage_config=preimage_config_identity_target_3,
        )
        assert preimage._preimage_config is preimage_config_identity_target_3

    def test_is_quantum_algorithm(self, preimage_instance):
        """Preimage is a QuantumAlgorithm subclass."""
        assert isinstance(preimage_instance, QuantumAlgorithm)


# === Tests: Preimage type checking ===


class TestPreimageTypeChecking:

    def test_none_executor_raises(self, preimage_config_identity_target_3):
        """None executor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            Preimage(executor=None, preimage_config=preimage_config_identity_target_3)

    def test_wrong_type_executor_raises(self, preimage_config_identity_target_3):
        """Non-CircuitExecutor executor raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            Preimage(
                executor="not_an_executor",
                preimage_config=preimage_config_identity_target_3,
            )

    def test_none_config_raises(self, aer_executor):
        """None preimage_config raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            Preimage(executor=aer_executor, preimage_config=None)

    def test_wrong_type_config_raises(self, aer_executor):
        """Non-PreimageConfig raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            Preimage(executor=aer_executor, preimage_config="not_a_config")


# === Tests: Preimage._build_grover_circuit ===


class TestPreimageBuildGroverCircuit:

    def test_returns_quantum_circuit(self, preimage_instance, identity_2bit_vbf):
        """_build_grover_circuit returns a QuantumCircuit."""
        unitary = identity_2bit_vbf.to_circuit()
        unitary_extended = QuantumCircuit(unitary.num_qubits + 1)
        unitary_extended.compose(
            unitary, qubits=range(unitary.num_qubits), inplace=True
        )
        grover = preimage_instance._build_grover_circuit(unitary_extended)
        assert isinstance(grover, QuantumCircuit)


# === Tests: Preimage.run ===


class TestPreimageRun:

    def test_returns_qaa_result(self, iterative_executor, identity_2bit_vbf):
        """run() returns a QAAResult instance."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=3,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert isinstance(result, PreimageResult)

    def test_finds_preimage_of_identity_function(
        self, iterative_executor, identity_2bit_vbf
    ):
        """For identity f(x)=x, preimage of 3 is 3."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=0b11,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert 0b11 in result.preimages

    def test_finds_preimage_of_xor_function(self, iterative_executor, xor_one_2bit_vbf):
        """For f(x) = x ^ 1, preimage of 0 is 1."""
        config = PreimageConfig(
            vbf=xor_one_2bit_vbf,
            vbf_image=0b00,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert 0b01 in result.preimages

    def test_verified_results_are_actual_preimages(
        self, iterative_executor, identity_2bit_vbf
    ):
        """All verified results satisfy f(x) == target image."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=0b10,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        for x in result.preimages:
            assert identity_2bit_vbf.eval(x) == 0b10

    def test_iterations_is_positive(self, iterative_executor, identity_2bit_vbf):
        """At least 1 iteration was performed."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=1,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert result.iterations >= 1

    def test_execution_result_stored(self, iterative_executor, identity_2bit_vbf):
        """ExecutionResult is stored in the result."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=0,
            success_prob=0.25,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert isinstance(result.execution_result, ExecutionResult)

    def test_finds_preimage_of_constant_zero(
        self, iterative_executor, constant_zero_2bit_vbf
    ):
        """For f(x)=0, any x is a preimage of 0."""
        config = PreimageConfig(
            vbf=constant_zero_2bit_vbf,
            vbf_image=0,
            success_prob=1.0,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert len(result.preimages) > 0
        for x in result.preimages:
            assert constant_zero_2bit_vbf.eval(x) == 0

    def test_unknown_probability_finds_preimage(
        self, iterative_executor, identity_2bit_vbf
    ):
        """run() with unknown success probability (None) still finds a preimage."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=0b10,
            success_prob=None,
            seed=42,
            base=1.5,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert 0b10 in result.preimages


# === Tests: Preimage reproducibility ===


class TestPreimageReproducibility:

    def test_same_seed_produces_same_result(self, identity_2bit_vbf):
        """Same seed produces identical verified results."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=1,
            success_prob=0.25,
            seed=42,
        )
        exec_config = ExecutionConfig(
            run_options={"shots": 1, "simulation_seed": 42}, skip_transpilation=True
        )

        exec1 = IterativeExecutor(AerExecutor(exec_config), base_seed=50)
        exec2 = IterativeExecutor(AerExecutor(exec_config), base_seed=50)

        result1 = Preimage(executor=exec1, preimage_config=config).run()
        result2 = Preimage(executor=exec2, preimage_config=config).run()

        assert result1.preimages == result2.preimages
        assert result1.iterations == result2.iterations

    def test_reset_executor_replays_result(self, identity_2bit_vbf):
        """Resetting the IterativeExecutor allows reproducible re-runs."""
        config = PreimageConfig(
            vbf=identity_2bit_vbf,
            vbf_image=3,
            success_prob=0.25,
            seed=42,
        )
        exec_config = ExecutionConfig(
            run_options={"shots": 1, "simulation_seed": 42}, skip_transpilation=True
        )
        executor = IterativeExecutor(AerExecutor(exec_config), base_seed=77)

        preimage = Preimage(executor=executor, preimage_config=config)
        result1 = preimage.run()
        executor.reset()
        result2 = preimage.run()

        assert result1.preimages == result2.preimages
        assert result1.iterations == result2.iterations


# === Tests: Preimage with different VBFs ===


class TestPreimageDifferentVBFs:

    def test_1bit_to_1bit_identity(self, iterative_executor):
        """Preimage search works on a 1-bit identity function."""
        vbf = VectorialBooleanFunctionFull(lambda x: x, 1, 1)
        config = PreimageConfig(
            vbf=vbf,
            vbf_image=1,
            success_prob=0.5,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        assert 0b1 in result.preimages

    def test_2bit_to_1bit_function(self, iterative_executor):
        """Preimage search works on a 2-bit to 1-bit function."""
        # f(0)=0, f(1)=1, f(2)=1, f(3)=0
        vbf = VectorialBooleanFunctionFull(lambda x: x & 1, 2, 1)
        config = PreimageConfig(
            vbf=vbf,
            vbf_image=1,
            success_prob=0.5,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        # Preimages of 1 are 1 and 3
        for x in result.preimages:
            assert vbf.eval(x) == 1

    def test_3bit_to_3bit_function(self, iterative_executor):
        """Preimage search works on a 3-bit to 3-bit function."""
        vbf = VectorialBooleanFunctionFull(lambda x: x ^ 0b101, 3, 3)
        config = PreimageConfig(
            vbf=vbf,
            vbf_image=0b101,
            success_prob=1.0 / 8,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()
        # f(x) = x ^ 5, so f(x)=5 when x=0
        assert 0 in result.preimages


class TestPreimageKeyRecovery:
    """Tests for Preimage simulating key recovery on XOR cipher.

    Scenario: f(key) = key ^ fixed_plaintext.
    We fix the plaintext and search for the key such that
    f(key) = target_ciphertext.
    """

    def test_2bit_key_recovery_ciphertext_zero(self, iterative_executor):
        """For 2-bit XOR cipher with fixed plaintext=0b10, find key such that key^plaintext=0."""
        m = 2
        fixed_plaintext = 0b10
        target_ciphertext = 0b00
        expected_key = fixed_plaintext ^ target_ciphertext  # 0b10

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert len(result.preimages) > 0
        assert expected_key in result.preimages

    def test_2bit_key_recovery_ciphertext_11(self, iterative_executor):
        """For 2-bit XOR cipher with fixed plaintext=0b01, find key such that key^plaintext=0b11."""
        m = 2
        fixed_plaintext = 0b01
        target_ciphertext = 0b11
        expected_key = fixed_plaintext ^ target_ciphertext  # 0b10

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert len(result.preimages) > 0
        assert expected_key in result.preimages

    def test_2bit_key_recovery_verifies_with_eval(self, iterative_executor):
        """All found keys satisfy key ^ fixed_plaintext == target_ciphertext."""
        m = 2
        fixed_plaintext = 0b11
        target_ciphertext = 0b01

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert len(result.preimages) > 0
        for key in result.preimages:
            assert vbf.eval(key) == target_ciphertext

    def test_2bit_key_recovery_unique_solution(self, iterative_executor):
        """XOR cipher key recovery yields exactly the unique correct key."""
        m = 2
        fixed_plaintext = 0b11
        target_ciphertext = 0b00
        expected_key = fixed_plaintext ^ target_ciphertext  # 0b11

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert result.preimages == [expected_key]

    def test_2bit_key_recovery_unknown_probability(self, iterative_executor):
        """Key recovery works with unknown success probability."""
        m = 2
        fixed_plaintext = 0b10
        target_ciphertext = 0b01
        expected_key = fixed_plaintext ^ target_ciphertext  # 0b11

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            seed=42,
            base=1.5,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert len(result.preimages) > 0
        assert expected_key in result.preimages

    def test_2bit_key_recovery_reproducible(self):
        """Same seed produces identical key recovery results."""
        m = 2
        fixed_plaintext = 0b01
        target_ciphertext = 0b10

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )

        exec_config = ExecutionConfig(run_options={"shots": 1}, skip_transpilation=True)

        exec1 = IterativeExecutor(AerExecutor(exec_config), base_seed=100)
        exec2 = IterativeExecutor(AerExecutor(exec_config), base_seed=100)

        result1 = Preimage(executor=exec1, preimage_config=config).run()
        result2 = Preimage(executor=exec2, preimage_config=config).run()

        assert result1.preimages == result2.preimages
        assert result1.iterations == result2.iterations

    def test_2bit_key_recovery_iterations_positive(self, iterative_executor):
        """Key recovery performs at least one iteration."""
        m = 2
        fixed_plaintext = 0b11
        target_ciphertext = 0b01

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert result.iterations >= 1

    def test_3bit_key_recovery_known_probability(self, iterative_executor):
        """Key recovery performs at least one iteration."""
        m = 3
        fixed_plaintext = 0b011
        target_ciphertext = 0b010

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            success_prob=1.0 / (1 << m),
            seed=43,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert 0b001 in result.preimages
        assert result.iterations == 2

    def test_3bit_key_recovery_unknown_probability(self, iterative_executor):
        """Key recovery performs at least one iteration."""
        m = 3
        fixed_plaintext = 0b011
        target_ciphertext = 0b010

        vbf = VectorialBooleanFunctionFull(lambda key: key ^ fixed_plaintext, m, m)

        config = PreimageConfig(
            vbf=vbf,
            vbf_image=target_ciphertext,
            seed=42,
        )
        preimage = Preimage(executor=iterative_executor, preimage_config=config)
        result = preimage.run()

        assert result.iterations == 1
        assert 0b001 in result.preimages
