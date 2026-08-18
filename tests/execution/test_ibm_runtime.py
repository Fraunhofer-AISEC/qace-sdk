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
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeNairobiV2

from qace.execution import (
    IBMRuntimeExecutor,
    IBMRuntimeExecutionConfig,
    ExecutionResult,
)

from qace.vbf import measure_msb0, measure_all_msb0

# === Fixtures ===


@pytest.fixture
def fake_backend():
    """Small 5-qubit fake backend."""
    return FakeManilaV2()


@pytest.fixture
def bell_circuit():
    """Simple 2-qubit Bell state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    measure_msb0(qc, [0, 1], [0, 1])
    return qc


@pytest.fixture
def bell_circuit_measure_all():
    """Bell state circuit using measure_all()."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    measure_all_msb0(qc)
    return qc


@pytest.fixture
def ghz_3_circuit():
    """3-qubit GHZ state circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    measure_all_msb0(qc)
    return qc


@pytest.fixture
def runtime_executor(fake_backend):
    """Default IBMRuntimeExecutor with fake backend."""
    config = IBMRuntimeExecutionConfig(backend_options={"backend": fake_backend})
    return IBMRuntimeExecutor(config=config)


@pytest.fixture
def seeded_executor(fake_backend):
    """IBMRuntimeExecutor with seed for reproducibility."""
    config = IBMRuntimeExecutionConfig(
        backend_options={"backend": fake_backend},
        run_options={"shots": 1024},
        sampler_options={"simulator": {"seed_simulator": 42}},
    )
    return IBMRuntimeExecutor(config=config)


# === Tests: IBMRuntimeExecutionConfig ===


class TestIBMRuntimeExecutionConfig:

    def test_default_sampler_options_empty(self):
        """Default sampler_options is an empty dict."""
        config = IBMRuntimeExecutionConfig()
        assert config.sampler_options == {}

    def test_sampler_options_stored(self):
        """Custom sampler_options are stored correctly."""
        config = IBMRuntimeExecutionConfig(
            sampler_options={"simulator": {"seed_simulator": 42}}
        )
        assert config.sampler_options == {"simulator": {"seed_simulator": 42}}

    def test_inherits_execution_config_fields(self):
        """IBMRuntimeExecutionConfig has all ExecutionConfig fields."""
        config = IBMRuntimeExecutionConfig(
            run_options={"shots": 2048},
            transpiler_options={"optimization_level": 2},
            backend_options={"backend": FakeManilaV2()},
            sampler_options={"simulator": {"seed_simulator": 1}},
        )
        assert config.run_options == {"shots": 2048}
        assert config.transpiler_options == {"optimization_level": 2}
        assert "backend" in config.backend_options
        assert "simulator" in config.sampler_options

    def test_sampler_options_not_shared_between_instances(self):
        """Each instance gets its own sampler_options dict."""
        config1 = IBMRuntimeExecutionConfig()
        config2 = IBMRuntimeExecutionConfig()
        config1.sampler_options["simulator"] = {"seed_simulator": 1}
        assert "simulator" not in config2.sampler_options


# === Tests: Initialization ===


class TestIBMRuntimeExecutorInit:

    def test_valid_backend_initializes(self, fake_backend):
        """Executor initializes with a valid BackendV2."""
        config = IBMRuntimeExecutionConfig(backend_options={"backend": fake_backend})
        executor = IBMRuntimeExecutor(config=config)
        assert isinstance(executor._backend, BackendV2)
        assert isinstance(executor._sampler, SamplerV2)

    def test_config_stored(self, fake_backend):
        """Config is stored on the executor."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 2048},
        )
        executor = IBMRuntimeExecutor(config=config)
        assert isinstance(executor.config.backend_options["backend"], FakeManilaV2)
        assert executor.config.run_options["shots"] is 2048

    def test_none_config_raises(self):
        """No config (empty backend_options) raises RuntimeError."""
        with pytest.raises(RuntimeError, match="backend"):
            IBMRuntimeExecutor(config=None)

    def test_empty_backend_options_raises(self):
        """Empty backend_options raises RuntimeError."""
        config = IBMRuntimeExecutionConfig(backend_options={})
        with pytest.raises(RuntimeError, match="backend"):
            IBMRuntimeExecutor(config=config)

    def test_missing_backend_key_raises(self):
        """backend_options without 'backend' key raises RuntimeError."""
        config = IBMRuntimeExecutionConfig(backend_options={"shots": 1024})
        with pytest.raises(RuntimeError, match="backend"):
            IBMRuntimeExecutor(config=config)

    def test_invalid_backend_type_string_raises(self):
        """String as backend raises RuntimeError."""
        config = IBMRuntimeExecutionConfig(backend_options={"backend": "not_a_backend"})
        with pytest.raises(RuntimeError, match="BackendV2"):
            IBMRuntimeExecutor(config=config)

    def test_invalid_backend_type_int_raises(self):
        """Integer as backend raises RuntimeError."""
        config = IBMRuntimeExecutionConfig(backend_options={"backend": 42})
        with pytest.raises(RuntimeError, match="BackendV2"):
            IBMRuntimeExecutor(config=config)

    def test_none_as_backend_value_raises(self):
        """None as backend value raises RuntimeError."""
        config = IBMRuntimeExecutionConfig(backend_options={"backend": None})
        with pytest.raises(RuntimeError, match="backend"):
            IBMRuntimeExecutor(config=config)

    def test_sampler_options_applied(self, fake_backend):
        """Sampler options are applied via update()."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            sampler_options={"simulator": {"seed_simulator": 123}},
        )
        executor = IBMRuntimeExecutor(config=config)
        assert executor._sampler.options.simulator.seed_simulator == 123


# === Tests: Execution ===


class TestIBMRuntimeExecutorExecute:

    def test_execute_returns_execution_result(self, runtime_executor, bell_circuit):
        """execute() returns an ExecutionResult instance."""
        result = runtime_executor.execute(bell_circuit)
        assert isinstance(result, ExecutionResult)

    def test_execute_counts_not_empty(self, runtime_executor, bell_circuit):
        """Result contains measurement counts."""
        result = runtime_executor.execute(bell_circuit)
        assert len(result.counts) > 0

    def test_execute_counts_valid_bitstrings(self, runtime_executor, bell_circuit):
        """Counts contain only valid 2-bit strings."""
        result = runtime_executor.execute(bell_circuit)
        for bitstring in result.counts.keys():
            cleaned = bitstring.replace(" ", "")
            assert all(c in "01" for c in cleaned)

    def test_execute_default_shots(self, runtime_executor, bell_circuit):
        """Default shot count is 1024."""
        result = runtime_executor.execute(bell_circuit)
        total = sum(result.counts.values())
        assert total == 1024

    def test_execute_custom_shots(self, fake_backend, bell_circuit):
        """Custom shot count is respected."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 500},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)
        total = sum(result.counts.values())
        assert total == 500

    def test_execute_high_shots(self, fake_backend, bell_circuit):
        """High shot count works correctly."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 8192},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)
        total = sum(result.counts.values())
        assert total == 8192

    def test_execute_measure_all_circuit(self, fake_backend, bell_circuit_measure_all):
        """Circuit using measure_all() works correctly."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit_measure_all)
        assert len(result.counts) > 0
        assert sum(result.counts.values()) == 1024

    def test_execute_3_qubit_circuit(self, fake_backend, ghz_3_circuit):
        """3-qubit GHZ circuit executes on 5-qubit backend."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(ghz_3_circuit)
        assert len(result.counts) > 0

    def test_execute_bell_state_has_correlated_outcomes(
        self, fake_backend, bell_circuit
    ):
        """Bell state produces predominantly '00' and '11'."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 10000},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)

        total = sum(result.counts.values())
        correlated = sum(
            v for k, v in result.counts.items() if k.replace(" ", "") in {"00", "11"}
        )
        assert correlated / total > 0.8


# === Tests: Reproducibility with seed ===


class TestIBMRuntimeExecutorReproducibility:

    def test_seeded_execution_reproducible(self, seeded_executor, bell_circuit):
        """Same seed produces identical results."""
        result1 = seeded_executor.execute(bell_circuit)
        result2 = seeded_executor.execute(bell_circuit)
        assert result1.counts == result2.counts

    def test_different_seeds_different_results(self, fake_backend, bell_circuit):
        """Different seeds produce different results."""
        config1 = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
            sampler_options={"simulator": {"seed_simulator": 1}},
        )
        config2 = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
            sampler_options={"simulator": {"seed_simulator": 9999}},
        )

        result1 = IBMRuntimeExecutor(config1).execute(bell_circuit)
        result2 = IBMRuntimeExecutor(config2).execute(bell_circuit)
        assert result1.counts != result2.counts

    def test_no_seed_may_vary(self, fake_backend, bell_circuit):
        """Without seed, results may vary (non-deterministic)."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
        )
        # Just verify it runs without error
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)
        assert len(result.counts) > 0


# === Tests: Noise from backend ===


class TestIBMRuntimeExecutorNoise:

    def test_noise_introduced(self, fake_backend, bell_circuit):
        """Fake backend introduces noise (not only '00' and '11')."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 10000},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)

        all_keys = {k.replace(" ", "") for k in result.counts.keys()}
        noisy_outcomes = all_keys - {"00", "11"}
        assert len(noisy_outcomes) > 0, "Expected noise from fake backend"

    def test_different_backends_different_results(self, bell_circuit):
        """Different fake backends produce different distributions."""
        config_manila = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeManilaV2()},
            run_options={"shots": 10000},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )
        config_nairobi = IBMRuntimeExecutionConfig(
            backend_options={"backend": FakeNairobiV2()},
            run_options={"shots": 10000},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )

        result_manila = IBMRuntimeExecutor(config_manila).execute(bell_circuit)
        result_nairobi = IBMRuntimeExecutor(config_nairobi).execute(bell_circuit)

        assert result_manila.counts != result_nairobi.counts


# === Tests: Metadata ===


class TestIBMRuntimeExecutorMetadata:

    def test_metadata_contains_timing(self, runtime_executor, bell_circuit):
        """Metadata includes all timing fields."""
        result = runtime_executor.execute(bell_circuit)

        assert "transpile_time_s" in result.metadata
        assert "execution_time_s" in result.metadata
        assert "total_time_s" in result.metadata

    def test_metadata_timing_positive(self, runtime_executor, bell_circuit):
        """All timing values are positive."""
        result = runtime_executor.execute(bell_circuit)

        assert result.metadata["transpile_time_s"] > 0
        assert result.metadata["execution_time_s"] > 0
        assert result.metadata["total_time_s"] > 0

    def test_metadata_total_time_is_sum(self, runtime_executor, bell_circuit):
        """total_time equals transpile_time plus execution_time."""
        result = runtime_executor.execute(bell_circuit)

        expected = (
            result.metadata["transpile_time_s"] + result.metadata["execution_time_s"]
        )
        assert result.metadata["total_time_s"] == pytest.approx(expected)

    def test_metadata_contains_circuit_info(self, runtime_executor, bell_circuit):
        """Metadata includes transpiled circuit info."""
        result = runtime_executor.execute(bell_circuit)

        assert result.metadata["transpiled_depth"] > 0
        assert result.metadata["transpiled_gate_count"] > 0


# === Tests: Transpiler options ===


class TestIBMRuntimeExecutorTranspiler:

    def test_optimization_level_respected(self, fake_backend, bell_circuit):
        """Different optimization levels produce different transpilations."""
        config_0 = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            transpiler_options={"optimization_level": 0, "seed_transpiler": 42},
            run_options={"shots": 1024},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )
        config_3 = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            transpiler_options={"optimization_level": 3, "seed_transpiler": 42},
            run_options={"shots": 1024},
            sampler_options={"simulator": {"seed_simulator": 42}},
        )

        result_0 = IBMRuntimeExecutor(config_0).execute(bell_circuit)
        result_3 = IBMRuntimeExecutor(config_3).execute(bell_circuit)

        assert (
            result_3.metadata["transpiled_depth"]
            <= result_0.metadata["transpiled_depth"]
        )

    def test_transpiler_options_not_mutated(self, fake_backend, bell_circuit):
        """Transpiler options in config are not mutated by execute()."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            transpiler_options={"optimization_level": 1},
        )
        executor = IBMRuntimeExecutor(config=config)
        executor.execute(bell_circuit)

        assert config.transpiler_options == {"optimization_level": 1}


# === Tests: Error handling ===


class TestIBMRuntimeExecutorErrorHandling:

    def test_circuit_without_measurements_raises(self, fake_backend):
        """Circuit without measurements raises RuntimeError."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        config = IBMRuntimeExecutionConfig(backend_options={"backend": fake_backend})
        executor = IBMRuntimeExecutor(config=config)

        with pytest.raises(RuntimeError):
            executor.execute(qc)

    def test_circuit_too_large_for_backend_raises(self, fake_backend):
        """Circuit with more qubits than backend raises error."""
        qc = QuantumCircuit(10, 10)
        for i in range(9):
            qc.cx(i, i + 1)
        measure_all_msb0(qc)

        config = IBMRuntimeExecutionConfig(backend_options={"backend": fake_backend})
        executor = IBMRuntimeExecutor(config=config)

        with pytest.raises(RuntimeError):
            executor.execute(qc)


# === Tests: Result object ===


class TestIBMRuntimeExecutorResult:

    def test_result_object_stored(self, runtime_executor, bell_circuit):
        """The raw PrimitiveResult is stored in result.result."""
        result = runtime_executor.execute(bell_circuit)
        assert result.result is not None

    def test_result_counts_are_dict(self, runtime_executor, bell_circuit):
        """Counts are a dictionary."""
        result = runtime_executor.execute(bell_circuit)
        assert isinstance(result.counts, dict)

    def test_result_counts_values_are_ints(self, runtime_executor, bell_circuit):
        """Count values are integers."""
        result = runtime_executor.execute(bell_circuit)
        for value in result.counts.values():
            assert isinstance(value, int)

    def test_result_counts_sum_equals_shots(self, fake_backend, bell_circuit):
        """Sum of counts equals requested shots."""
        shots = 2048
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": shots},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)
        assert sum(result.counts.values()) == shots


# === Tests: Sampler options ===


class TestIBMRuntimeExecutorSamplerOptions:

    def test_empty_sampler_options_works(self, fake_backend, bell_circuit):
        """Empty sampler_options does not cause errors."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            sampler_options={},
        )
        executor = IBMRuntimeExecutor(config=config)
        result = executor.execute(bell_circuit)
        assert len(result.counts) > 0

    def test_seed_simulator_applied(self, fake_backend, bell_circuit):
        """seed_simulator in sampler_options makes execution deterministic."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1024},
            sampler_options={"simulator": {"seed_simulator": 777}},
        )
        executor = IBMRuntimeExecutor(config=config)
        result1 = executor.execute(bell_circuit)
        result2 = executor.execute(bell_circuit)
        assert result1.counts == result2.counts

    def test_invalid_sampler_options_raises(self, fake_backend):
        """Invalid sampler options raise RuntimeError."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            sampler_options={"nonexistent_option": True},
        )
        with pytest.raises(RuntimeError, match="Failed to initialize SamplerV2"):
            IBMRuntimeExecutor(config=config)


# === Tests: skip_transpilation ===


class TestIBMRuntimeExecutorSkipTranspilation:

    def test_skip_with_pre_transpiled(self, fake_backend, bell_circuit):
        pre_transpiled = qk_transpile(
            bell_circuit, backend=fake_backend, seed_transpiler=42
        )

        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
            sampler_options={"simulator": {"seed_simulator": 42}},
            skip_transpilation=True,
        )
        result = IBMRuntimeExecutor(config=config).execute(pre_transpiled)

        assert sum(result.counts.values()) == 100
        assert result.metadata["transpile_time_s"] == 0.0
