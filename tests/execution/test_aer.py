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
from qiskit import QuantumCircuit
from qiskit import transpile as qk_transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qace.execution import *
from qace.vbf import measure_all_msb0

# === Fixtures ===


@pytest.fixture
def default_config():
    return ExecutionConfig()


@pytest.fixture
def bell_circuit():
    """Simple 2-qubit Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    measure_all_msb0(qc)
    return qc


@pytest.fixture
def aer_executor(default_config):
    return AerExecutor(config=default_config)


class TestAerExecutorInit:

    def test_default_config(self):
        """Executor can be created with default config."""
        executor = AerExecutor()
        assert executor.config is not None
        assert isinstance(executor._backend, AerSimulator)

    def test_custom_backend_options(self):
        """Executor accepts valid backend_options."""
        config = ExecutionConfig(
            backend_options={"method": "statevector", "seed_simulator": 42}
        )
        executor = AerExecutor(config=config)
        assert executor._backend is not None

    def test_invalid_backend_options_raises_aer_error(self):
        """Invalid backend_options raise RuntimeError."""
        config = ExecutionConfig(backend_options={"method": "nonexistent_method"})
        with pytest.raises(RuntimeError, match="Failed to initialize AerSimulator"):
            AerExecutor(config=config)


# === Tests: execute() ===


class TestAerExecutorExecute:

    def test_execute_returns_execution_result(self, aer_executor, bell_circuit):
        """execute() returns an ExecutionResult instance."""
        result = aer_executor.execute(bell_circuit)
        assert isinstance(result, ExecutionResult)

    def test_execute_counts_not_empty(self, aer_executor, bell_circuit):
        """Result contains measurement counts."""
        result = aer_executor.execute(bell_circuit)
        assert len(result.counts) > 0

    def test_execute_counts_valid_bitstrings(self, aer_executor, bell_circuit):
        """Counts contain only valid 2-bit strings."""
        result = aer_executor.execute(bell_circuit)
        for bitstring in result.counts.keys():
            assert all(c in "01 " for c in bitstring)

    def test_execute_bell_state_only_correlated_outcomes(self, bell_circuit):
        """Bell state yields only '00' and '11'."""
        config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            transpiler_options={"seed_transpiler": 42},
        )
        executor = AerExecutor(config=config)
        result = executor.execute(bell_circuit)

        valid_outcomes = {"00", "11"}
        for bitstring in result.counts.keys():
            cleaned = bitstring.replace(" ", "")
            assert cleaned in valid_outcomes

    def test_execute_shots_respected(self, bell_circuit):
        """Number of shots is correctly applied."""
        config = ExecutionConfig(run_options={"shots": 100})
        executor = AerExecutor(config=config)
        result = executor.execute(bell_circuit)

        total_counts = sum(result.counts.values())
        assert total_counts == 100

    def test_execute_reproducible_with_seed(self, bell_circuit):
        """Same seed produces identical results."""
        config = ExecutionConfig(
            backend_options={"seed_simulator": 12345},
            run_options={"shots": 1024},
        )
        executor = AerExecutor(config=config)

        result1 = executor.execute(bell_circuit)
        result2 = executor.execute(bell_circuit)
        assert result1.counts == result2.counts


# === Tests: Metadata ===


class TestAerExecutorMetadata:

    def test_metadata_contains_timing(self, aer_executor, bell_circuit):
        """Metadata includes timing information."""
        result = aer_executor.execute(bell_circuit)

        assert "transpile_time_s" in result.metadata
        assert "execution_time_s" in result.metadata
        assert "total_time_s" in result.metadata

    def test_metadata_timing_positive(self, aer_executor, bell_circuit):
        """All timing values are positive."""
        result = aer_executor.execute(bell_circuit)

        assert result.metadata["transpile_time_s"] > 0
        assert result.metadata["execution_time_s"] > 0
        assert result.metadata["total_time_s"] > 0

    def test_metadata_total_time_is_sum(self, aer_executor, bell_circuit):
        """total_time equals transpile_time plus execution_time."""
        result = aer_executor.execute(bell_circuit)

        expected = (
            result.metadata["transpile_time_s"] + result.metadata["execution_time_s"]
        )
        assert result.metadata["total_time_s"] == pytest.approx(expected)

    def test_metadata_contains_circuit_info(self, aer_executor, bell_circuit):
        """Metadata includes transpiled circuit information."""
        result = aer_executor.execute(bell_circuit)

        assert "transpiled_depth" in result.metadata
        assert "transpiled_gate_count" in result.metadata
        assert result.metadata["transpiled_depth"] > 0
        assert result.metadata["transpiled_gate_count"] > 0


# === Tests: Noisy Backend ===


class TestAerExecutorNoisy:

    @pytest.fixture
    def simple_noise_model(self):
        """Simple depolarizing noise model."""
        noise_model = NoiseModel()
        # Single-qubit error for 'h'
        error_1q = depolarizing_error(0.01, 1)
        # Two-qubit error for 'cx'
        error_2q = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        return noise_model

    @pytest.fixture
    def noisy_executor(self, simple_noise_model):
        """Executor configured with a noise model."""
        config = ExecutionConfig(
            backend_options={
                "noise_model": simple_noise_model,
                "seed_simulator": 42,
            },
            run_options={"shots": 4096},
        )
        return AerExecutor(config=config)

    def test_noisy_executor_initializes(self, noisy_executor):
        """Executor with noise model initializes without error."""
        assert noisy_executor._backend is not None

    def test_noisy_execution_returns_result(self, noisy_executor, bell_circuit):
        """Noisy execution returns a valid ExecutionResult."""
        result = noisy_executor.execute(bell_circuit)
        assert isinstance(result, ExecutionResult)
        assert len(result.counts) > 0

    def test_noisy_execution_introduces_errors(self, bell_circuit, simple_noise_model):
        """Noisy simulation produces outcomes beyond ideal '00' and '11'."""
        config = ExecutionConfig(
            backend_options={
                "noise_model": simple_noise_model,
                "seed_simulator": 123,
            },
            run_options={"shots": 10000},
        )
        executor = AerExecutor(config=config)
        result = executor.execute(bell_circuit)

        # With noise, we expect some '01' or '10' outcomes
        all_bitstrings = {k.replace(" ", "") for k in result.counts.keys()}
        noisy_outcomes = all_bitstrings - {"00", "11"}
        assert len(noisy_outcomes) > 0, "Expected noise to introduce erroneous outcomes"

    def test_noisy_vs_ideal_different_distributions(
        self, bell_circuit, simple_noise_model
    ):
        """Noisy and ideal execution produce different count distributions."""
        ideal_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 4096},
        )
        noisy_config = ExecutionConfig(
            backend_options={
                "noise_model": simple_noise_model,
                "seed_simulator": 42,
            },
            run_options={"shots": 4096},
        )

        ideal_result = AerExecutor(ideal_config).execute(bell_circuit)
        noisy_result = AerExecutor(noisy_config).execute(bell_circuit)

        assert ideal_result.counts != noisy_result.counts

    def test_noisy_execution_reproducible_with_seed(
        self, bell_circuit, simple_noise_model
    ):
        """Same seed produces identical noisy results."""
        config = ExecutionConfig(
            backend_options={
                "noise_model": simple_noise_model,
                "seed_simulator": 777,
            },
            run_options={"shots": 2048},
        )

        executor = AerExecutor(config=config)
        result1 = executor.execute(bell_circuit)
        result2 = executor.execute(bell_circuit)
        assert result1.counts == result2.counts

    def test_high_noise_degrades_fidelity(self, bell_circuit):
        """Higher noise leads to more erroneous outcomes."""
        low_noise = NoiseModel()
        low_noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["h"])
        low_noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 2), ["cx"])

        high_noise = NoiseModel()
        high_noise.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ["h"])
        high_noise.add_all_qubit_quantum_error(depolarizing_error(0.1, 2), ["cx"])

        low_config = ExecutionConfig(
            backend_options={"noise_model": low_noise, "seed_simulator": 42},
            run_options={"shots": 10000},
        )
        high_config = ExecutionConfig(
            backend_options={"noise_model": high_noise, "seed_simulator": 42},
            run_options={"shots": 10000},
        )

        low_result = AerExecutor(low_config).execute(bell_circuit)
        high_result = AerExecutor(high_config).execute(bell_circuit)

        # Count erroneous outcomes (not '00' or '11')
        def error_count(counts):
            return sum(
                v for k, v in counts.items() if k.replace(" ", "") not in {"00", "11"}
            )

        assert error_count(high_result.counts) > error_count(low_result.counts)

    def test_noise_model_passed_to_backend(self, simple_noise_model):
        """Noise model is correctly passed to the AerSimulator backend."""
        config = ExecutionConfig(backend_options={"noise_model": simple_noise_model})
        executor = AerExecutor(config=config)

        backend_options = executor._backend.options
        assert backend_options.noise_model == simple_noise_model

    def test_different_noise_models_different_results(self, bell_circuit):
        """Different noise models produce different distributions."""
        noise_a = NoiseModel()
        noise_a.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["h"])

        noise_b = NoiseModel()
        noise_b.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])

        config_a = ExecutionConfig(
            backend_options={"noise_model": noise_a, "seed_simulator": 42},
            run_options={"shots": 8192},
        )
        config_b = ExecutionConfig(
            backend_options={"noise_model": noise_b, "seed_simulator": 42},
            run_options={"shots": 8192},
        )

        result_a = AerExecutor(config_a).execute(bell_circuit)
        result_b = AerExecutor(config_b).execute(bell_circuit)

        assert result_a.counts != result_b.counts


# === Fixtures ===
@pytest.fixture
def fake_backend():
    """Small 5-qubit fake backend (FakeManilaV2)."""
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    return FakeManilaV2()


@pytest.fixture
def ghz_3_circuit():
    """3-qubit GHZ state circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    measure_all_msb0(qc)
    return qc


class TestAerBackendV2ExecutorInit:

    def test_valid_backend_initializes(self, fake_backend):
        """Executor initializes with a valid BackendV2."""
        config = ExecutionConfig(backend_options={"backend": fake_backend})
        executor = AerBackendV2Executor(config=config)
        assert executor._backend is not None
        assert isinstance(executor._backend, AerSimulator)

    def test_valid_backend_with_simulator_options(self, fake_backend):
        """Executor accepts additional simulator options."""
        config = ExecutionConfig(
            backend_options={
                "backend": fake_backend,
                "seed_simulator": 42,
            }
        )
        executor = AerBackendV2Executor(config=config)
        assert executor._backend is not None

    def test_none_config_raises(self):
        """No config (empty backend_options) raises RuntimeError."""
        with pytest.raises(RuntimeError, match="backend"):
            AerBackendV2Executor(config=None)

    def test_empty_backend_options_raises(self):
        """Empty backend_options raises RuntimeError."""
        config = ExecutionConfig(backend_options={})
        with pytest.raises(RuntimeError, match="backend"):
            AerBackendV2Executor(config=config)

    def test_missing_backend_key_raises(self):
        """backend_options without 'backend' key raises RuntimeError."""
        config = ExecutionConfig(backend_options={"method": "statevector"})
        with pytest.raises(RuntimeError, match="backend"):
            AerBackendV2Executor(config=config)

    def test_invalid_backend_type_raises(self):
        """Non-BackendV2 value raises RuntimeError."""
        config = ExecutionConfig(backend_options={"backend": "not_a_backend"})
        with pytest.raises(RuntimeError, match="BackendV2"):
            AerBackendV2Executor(config=config)

    def test_integer_as_backend_raises(self):
        """Integer as backend raises RuntimeError."""
        config = ExecutionConfig(backend_options={"backend": 42})
        with pytest.raises(RuntimeError, match="BackendV2"):
            AerBackendV2Executor(config=config)

    def test_none_as_backend_raises(self):
        """None as backend value raises RuntimeError."""
        config = ExecutionConfig(backend_options={"backend": None})
        with pytest.raises(RuntimeError, match="backend"):
            AerBackendV2Executor(config=config)


class TestAerBackendV2ExecutorExecute:

    def test_execute_returns_result(self, fake_backend, bell_circuit):
        """execute() returns an ExecutionResult."""
        config = ExecutionConfig(backend_options={"backend": fake_backend})
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)
        assert isinstance(result, ExecutionResult)

    def test_execute_counts_not_empty(self, fake_backend, bell_circuit):
        """Result contains measurement counts."""
        config = ExecutionConfig(backend_options={"backend": fake_backend})
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)
        assert len(result.counts) > 0

    def test_execute_respects_shots(self, fake_backend, bell_circuit):
        """Shot count is respected."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 500},
        )
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)
        total = sum(result.counts.values())
        assert total == 500

    def test_execute_3_qubit_circuit(self, fake_backend, ghz_3_circuit):
        """3-qubit GHZ circuit executes on 5-qubit fake backend."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
        )
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(ghz_3_circuit)
        assert len(result.counts) > 0

    def test_execute_reproducible_with_seed(self, fake_backend, bell_circuit):
        """Same seed produces identical results."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend, "seed_simulator": 12345},
            run_options={"shots": 1024},
        )
        executor = AerBackendV2Executor(config=config)
        result1 = executor.execute(bell_circuit)
        result2 = executor.execute(bell_circuit)
        assert result1.counts == result2.counts


# === Tests: Noise from real backend ===


class TestAerBackendV2ExecutorNoise:

    def test_noise_introduced(self, fake_backend, bell_circuit):
        """Fake backend introduces noise (not only '00' and '11')."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend, "seed_simulator": 42},
            run_options={"shots": 10000},
        )
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)

        all_keys = {k.replace(" ", "") for k in result.counts.keys()}
        noisy_outcomes = all_keys - {"00", "11"}
        assert len(noisy_outcomes) > 0, "Expected noise from fake backend"

    def test_noisier_than_ideal(self, fake_backend, bell_circuit):
        """Fake backend produces more errors than ideal simulation."""
        from qace.execution import AerExecutor

        # Ideal
        ideal_config = ExecutionConfig(
            backend_options={"seed_simulator": 42},
            run_options={"shots": 10000},
        )
        ideal_result = AerExecutor(ideal_config).execute(bell_circuit)

        # Noisy (fake backend)
        noisy_config = ExecutionConfig(
            backend_options={"backend": fake_backend, "seed_simulator": 42},
            run_options={"shots": 10000},
        )
        noisy_result = AerBackendV2Executor(noisy_config).execute(bell_circuit)

        def error_count(counts):
            return sum(
                v for k, v in counts.items() if k.replace(" ", "") not in {"00", "11"}
            )

        assert error_count(noisy_result.counts) > error_count(ideal_result.counts)

    def test_different_backends_different_noise(self, bell_circuit):
        """Different fake backends produce different noise profiles."""
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeNairobiV2

        config_manila = ExecutionConfig(
            backend_options={"backend": FakeManilaV2(), "seed_simulator": 42},
            run_options={"shots": 10000},
        )
        config_nairobi = ExecutionConfig(
            backend_options={"backend": FakeNairobiV2(), "seed_simulator": 42},
            run_options={"shots": 10000},
        )

        result_manila = AerBackendV2Executor(config_manila).execute(bell_circuit)
        result_nairobi = AerBackendV2Executor(config_nairobi).execute(bell_circuit)

        assert result_manila.counts != result_nairobi.counts


# === Tests: Metadata ===


class TestAerBackendV2ExecutorMetadata:

    def test_metadata_contains_timing(self, fake_backend, bell_circuit):
        """Metadata includes all timing fields."""
        config = ExecutionConfig(backend_options={"backend": fake_backend})
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)

        assert "transpile_time_s" in result.metadata
        assert "execution_time_s" in result.metadata
        assert "total_time_s" in result.metadata

    def test_metadata_contains_circuit_info(self, fake_backend, bell_circuit):
        """Metadata includes transpiled circuit info."""
        config = ExecutionConfig(backend_options={"backend": fake_backend})
        executor = AerBackendV2Executor(config=config)
        result = executor.execute(bell_circuit)

        assert result.metadata["transpiled_depth"] > 0
        assert result.metadata["transpiled_gate_count"] > 0


# === Tests: Transpiler options ===


class TestAerBackendV2ExecutorTranspiler:

    def test_optimization_level_respected(self, fake_backend, bell_circuit):
        """Different optimization levels produce different transpilations."""
        config_0 = ExecutionConfig(
            backend_options={"backend": fake_backend, "seed_simulator": 42},
            transpiler_options={"optimization_level": 0, "seed_transpiler": 42},
            run_options={"shots": 1024},
        )
        config_3 = ExecutionConfig(
            backend_options={"backend": fake_backend, "seed_simulator": 42},
            transpiler_options={"optimization_level": 3, "seed_transpiler": 42},
            run_options={"shots": 1024},
        )

        result_0 = AerBackendV2Executor(config_0).execute(bell_circuit)
        result_3 = AerBackendV2Executor(config_3).execute(bell_circuit)

        # Higher optimization should produce equal or lower depth
        assert (
            result_3.metadata["transpiled_depth"]
            <= result_0.metadata["transpiled_depth"]
        )


# === Tests: Execution Error Handling ===


class TestAerExecutorExecutionErrors:

    def test_invalid_run_option_raises_aer_error(self, bell_circuit):
        """Invalid run_options raise RuntimeError."""
        config = ExecutionConfig(run_options={"shots": -1})
        executor = AerExecutor(config=config)

        with pytest.raises(RuntimeError, match="Failed to execute"):
            executor.execute(bell_circuit)

    def test_invalid_memory_option_raises_aer_error(self, bell_circuit):
        """Invalid run option value raises RuntimeError."""
        config = ExecutionConfig(run_options={"memory_slots": -5})
        executor = AerExecutor(config=config)

        with pytest.raises(RuntimeError, match="Failed to execute"):
            executor.execute(bell_circuit)


class TestAerBackendV2ExecutorExecutionErrors:

    def test_invalid_run_option_raises_aer_error(self, fake_backend, bell_circuit):
        """Invalid run_options raise RuntimeError."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": -1},
        )
        executor = AerBackendV2Executor(config=config)

        with pytest.raises(RuntimeError, match="Failed to execute"):
            executor.execute(bell_circuit)

    def test_circuit_too_large_for_backend_raises(self, fake_backend):
        """Circuit with more qubits than backend raises error."""
        # FakeManilaV2 has 5 qubits
        qc = QuantumCircuit(10)
        for i in range(9):
            qc.cx(i, i + 1)
        measure_all_msb0(qc)

        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
        )
        executor = AerBackendV2Executor(config=config)

        with pytest.raises(RuntimeError, match="Failed to transpile"):
            executor.execute(qc)

    def test_transpile_error_with_invalid_coupling_map(self, fake_backend):
        """Invalid transpiler options raise RuntimeError."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        measure_all_msb0(qc)

        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            transpiler_options={"coupling_map": [[0, 1]]},
        )
        executor = AerBackendV2Executor(config=config)

        with pytest.raises(RuntimeError, match="Failed to transpile"):
            executor.execute(qc)


# === Tests: skip_transpilation ===


class TestAerExecutorSkipTranspilation:

    def test_skip_executes_and_metadata_correct(self, bell_circuit):
        config = ExecutionConfig(
            run_options={"shots": 100},
            backend_options={"seed_simulator": 42},
            skip_transpilation=True,
        )
        result = AerExecutor(config=config).execute(bell_circuit)

        assert sum(result.counts.values()) == 100
        assert result.metadata["transpile_time_s"] == 0.0
        assert result.metadata["transpiled_depth"] == bell_circuit.depth()


class TestAerBackendV2ExecutorSkipTranspilation:

    def test_skip_with_pre_transpiled(self, fake_backend, bell_circuit):
        pre_transpiled = qk_transpile(
            bell_circuit, backend=fake_backend, seed_transpiler=42
        )

        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
            skip_transpilation=True,
        )
        result = AerBackendV2Executor(config=config).execute(pre_transpiled)

        assert sum(result.counts.values()) == 100
        assert result.metadata["transpile_time_s"] == 0.0
