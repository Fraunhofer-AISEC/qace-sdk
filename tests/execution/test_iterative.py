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
"""Tests for the IterativeExecutor wrapper."""

import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from qace.execution import (
    AerExecutor,
    AerBackendV2Executor,
    IBMRuntimeExecutor,
    IBMRuntimeExecutionConfig,
    ExecutionConfig,
    ExecutionResult,
    IterativeExecutor,
    CircuitExecutor,
)

from qace.vbf import measure_all_msb0

# === Fixtures ===


@pytest.fixture
def bell_circuit():
    """2-qubit Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    measure_all_msb0(qc)
    return qc


@pytest.fixture
def aer_config():
    """Standard config with 100 shots, no fixed seed."""
    return ExecutionConfig(run_options={"shots": 100})


@pytest.fixture
def fake_backend():
    """5-qubit fake backend."""
    return FakeManilaV2()


# === Tests: Initialization ===


class TestInit:

    def test_is_circuit_executor(self, aer_config):
        """IterativeExecutor satisfies the CircuitExecutor interface."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=0)
        assert isinstance(executor, CircuitExecutor)

    def test_stores_inner_executor(self, aer_config):
        """The wrapped executor is accessible via .inner."""
        inner = AerExecutor(aer_config)
        executor = IterativeExecutor(inner, base_seed=0)
        assert executor.inner is inner

    def test_stores_base_seed(self, aer_config):
        """The configured base_seed is accessible."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=123)
        assert executor.base_seed == 123

    def test_default_base_seed_is_zero(self, aer_config):
        """base_seed defaults to 0 when omitted."""
        executor = IterativeExecutor(AerExecutor(aer_config))
        assert executor.base_seed == 0

    def test_starts_with_zero_calls(self, aer_config):
        """A fresh instance has call_count == 0."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        assert executor.call_count == 0


# === Tests: Properties are read-only ===


class TestReadOnlyProperties:

    def test_inner_not_writable(self, aer_config):
        """Assigning to .inner raises AttributeError."""
        executor = IterativeExecutor(AerExecutor(aer_config))
        with pytest.raises(AttributeError):
            executor.inner = None

    def test_base_seed_not_writable(self, aer_config):
        """Assigning to .base_seed raises AttributeError."""
        executor = IterativeExecutor(AerExecutor(aer_config))
        with pytest.raises(AttributeError):
            executor.base_seed = 99

    def test_call_count_not_writable(self, aer_config):
        """Assigning to .call_count raises AttributeError."""
        executor = IterativeExecutor(AerExecutor(aer_config))
        with pytest.raises(AttributeError):
            executor.call_count = 5


# === Tests: Call counting ===


class TestCallCounting:

    def test_increments_per_call(self, aer_config, bell_circuit):
        """call_count increases by 1 with each execute()."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)

        executor.execute(bell_circuit)
        assert executor.call_count == 1

        executor.execute(bell_circuit)
        assert executor.call_count == 2

    def test_increments_on_failed_execution(self, bell_circuit):
        """call_count increases even when execute() raises."""
        config = ExecutionConfig(run_options={"shots": -1})
        executor = IterativeExecutor(AerExecutor(config), base_seed=42)

        with pytest.raises(RuntimeError):
            executor.execute(bell_circuit)

        assert executor.call_count == 1


# === Tests: Reset ===


class TestReset:

    def test_sets_call_count_to_zero(self, aer_config, bell_circuit):
        """reset() restores call_count to 0."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        executor.execute(bell_circuit)
        executor.execute(bell_circuit)

        executor.reset()

        assert executor.call_count == 0

    def test_is_idempotent(self, aer_config):
        """Calling reset() multiple times has no additional effect."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        executor.reset()
        executor.reset()
        assert executor.call_count == 0


# === Tests: Execution produces valid results ===


class TestExecutionResults:

    def test_returns_execution_result(self, aer_config, bell_circuit):
        """execute() returns an ExecutionResult instance."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        result = executor.execute(bell_circuit)
        assert isinstance(result, ExecutionResult)

    def test_shot_count_matches_config(self, aer_config, bell_circuit):
        """Total measurement count equals the configured shots."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        result = executor.execute(bell_circuit)
        assert sum(result.counts.values()) == 100

    def test_metadata_forwarded_from_inner(self, aer_config, bell_circuit):
        """Metadata from the inner executor is present in the result."""
        executor = IterativeExecutor(AerExecutor(aer_config), base_seed=42)
        result = executor.execute(bell_circuit)
        assert "transpiled_depth" in result.metadata
        assert "execution_time_s" in result.metadata


# === Tests: Seed injection – correct values ===


class TestSeedInjection:

    def test_first_call_uses_base_seed(self, bell_circuit):
        """First execute() uses base_seed as the simulator seed."""
        config = ExecutionConfig(run_options={"shots": 100})

        iterative = IterativeExecutor(AerExecutor(config), base_seed=42)
        r_iterative = iterative.execute(bell_circuit)

        direct_config = ExecutionConfig(
            run_options={"shots": 100, "seed_simulator": 42}
        )
        r_direct = AerExecutor(direct_config).execute(bell_circuit)

        assert r_iterative.counts == r_direct.counts

    def test_second_call_uses_base_seed_plus_one(self, bell_circuit):
        """Second execute() uses base_seed + 1 as the simulator seed."""
        config = ExecutionConfig(run_options={"shots": 100})

        iterative = IterativeExecutor(AerExecutor(config), base_seed=42)
        iterative.execute(bell_circuit)
        r_iterative = iterative.execute(bell_circuit)

        direct_config = ExecutionConfig(
            run_options={"shots": 100, "seed_simulator": 43}
        )
        r_direct = AerExecutor(direct_config).execute(bell_circuit)

        assert r_iterative.counts == r_direct.counts

    def test_nth_call_uses_base_seed_plus_n(self, bell_circuit):
        """The Nth call (0-indexed) uses base_seed + N as the simulator seed."""
        config = ExecutionConfig(run_options={"shots": 100})
        base = 100

        iterative = IterativeExecutor(AerExecutor(config), base_seed=base)

        for i in range(5):
            r_iterative = iterative.execute(bell_circuit)

            direct_config = ExecutionConfig(
                run_options={"shots": 100, "seed_simulator": base + i}
            )
            r_direct = AerExecutor(direct_config).execute(bell_circuit)

            assert r_iterative.counts == r_direct.counts


# === Tests: Consecutive calls produce different results ===


class TestConsecutiveCallsDiffer:

    def test_different_distributions_per_call(self, bell_circuit):
        """Successive execute() calls yield different count distributions."""
        config = ExecutionConfig(run_options={"shots": 1000})
        executor = IterativeExecutor(AerExecutor(config), base_seed=42)

        r1 = executor.execute(bell_circuit)
        r2 = executor.execute(bell_circuit)

        assert r1.counts != r2.counts

    def test_single_shot_produces_varied_outcomes(self, bell_circuit):
        """With shots=1, different seeds yield both possible Bell outcomes."""
        config = ExecutionConfig(run_options={"shots": 1})
        executor = IterativeExecutor(AerExecutor(config), base_seed=0)

        outcomes = set()
        for _ in range(50):
            result = executor.execute(bell_circuit)
            outcome = list(result.counts.keys())[0].replace(" ", "")
            outcomes.add(outcome)

        assert "00" in outcomes
        assert "11" in outcomes


# === Tests: Reproducibility ===


class TestReproducibility:

    def test_same_seed_same_sequence(self, bell_circuit):
        """Two executors with the same base_seed produce identical sequences."""
        config = ExecutionConfig(run_options={"shots": 100})

        exec1 = IterativeExecutor(AerExecutor(config), base_seed=7)
        exec2 = IterativeExecutor(AerExecutor(config), base_seed=7)

        for _ in range(5):
            r1 = exec1.execute(bell_circuit)
            r2 = exec2.execute(bell_circuit)
            assert r1.counts == r2.counts

    def test_different_seeds_different_sequence(self, bell_circuit):
        """Two executors with different base_seeds produce different results."""
        config = ExecutionConfig(run_options={"shots": 1000})

        exec1 = IterativeExecutor(AerExecutor(config), base_seed=1)
        exec2 = IterativeExecutor(AerExecutor(config), base_seed=9999)

        r1 = exec1.execute(bell_circuit)
        r2 = exec2.execute(bell_circuit)

        assert r1.counts != r2.counts

    def test_reset_replays_entire_sequence(self, bell_circuit):
        """After reset(), the full sequence of results is reproduced."""
        config = ExecutionConfig(run_options={"shots": 100})
        executor = IterativeExecutor(AerExecutor(config), base_seed=42)

        first_run = [executor.execute(bell_circuit).counts for _ in range(3)]

        executor.reset()

        second_run = [executor.execute(bell_circuit).counts for _ in range(3)]

        assert first_run == second_run


# === Tests: Inner executor is not permanently modified ===


class TestInnerExecutorIntegrity:

    def test_config_run_options_unchanged(self, aer_config, bell_circuit):
        """The inner executor's config.run_options is never mutated."""
        inner = AerExecutor(aer_config)
        executor = IterativeExecutor(inner, base_seed=42)

        executor.execute(bell_circuit)
        executor.execute(bell_circuit)

        assert "seed_simulator" not in aer_config.run_options

    def test_inner_usable_directly_after_iterative_use(self, bell_circuit):
        """The inner executor works normally after being used via IterativeExecutor."""
        config = ExecutionConfig(run_options={"shots": 100, "seed_simulator": 99})
        inner = AerExecutor(config)
        executor = IterativeExecutor(inner, base_seed=42)

        executor.execute(bell_circuit)
        executor.execute(bell_circuit)

        # Inner executor should behave as if nothing happened (seed=99)
        r1 = inner.execute(bell_circuit)
        r2 = inner.execute(bell_circuit)
        assert r1.counts == r2.counts

    def test_overrides_existing_seed_during_iterative_use(self, bell_circuit):
        """IterativeExecutor overrides any fixed seed_simulator in run_options."""
        config = ExecutionConfig(run_options={"shots": 100, "seed_simulator": 999})
        inner = AerExecutor(config)
        executor = IterativeExecutor(inner, base_seed=42)

        r1 = executor.execute(bell_circuit)
        r2 = executor.execute(bell_circuit)

        assert r1.counts != r2.counts


# === Tests: Error handling ===


class TestErrorHandling:

    def test_inner_exceptions_propagate(self, bell_circuit):
        """Exceptions from the inner executor propagate unchanged."""
        config = ExecutionConfig(run_options={"shots": -1})
        executor = IterativeExecutor(AerExecutor(config), base_seed=42)

        with pytest.raises(RuntimeError):
            executor.execute(bell_circuit)

    def test_usable_after_error(self, bell_circuit):
        """The executor works correctly after a previous error."""
        config = ExecutionConfig(run_options={"shots": 100})
        inner = AerExecutor(config)
        executor = IterativeExecutor(inner, base_seed=42)

        # Force error
        inner.config.run_options["shots"] = -1
        with pytest.raises(RuntimeError):
            executor.execute(bell_circuit)

        # Restore and continue
        inner.config.run_options["shots"] = 100
        result = executor.execute(bell_circuit)

        assert sum(result.counts.values()) == 100


# === Tests: Works with AerBackendV2Executor ===


class TestWithAerBackendV2:

    def test_produces_valid_result(self, fake_backend, bell_circuit):
        """Wrapping AerBackendV2Executor produces valid results."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
        )
        executor = IterativeExecutor(AerBackendV2Executor(config), base_seed=42)

        result = executor.execute(bell_circuit)

        assert isinstance(result, ExecutionResult)
        assert sum(result.counts.values()) == 100

    def test_reproducible_across_instances(self, fake_backend, bell_circuit):
        """Same base_seed gives identical results with BackendV2."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
            transpiler_options={"seed_transpiler": 42},
        )

        exec1 = IterativeExecutor(AerBackendV2Executor(config), base_seed=7)
        exec2 = IterativeExecutor(AerBackendV2Executor(config), base_seed=7)

        assert exec1.execute(bell_circuit).counts == exec2.execute(bell_circuit).counts

    def test_consecutive_calls_differ(self, fake_backend, bell_circuit):
        """Successive calls yield different distributions with BackendV2."""
        config = ExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1000},
            transpiler_options={"seed_transpiler": 42},
        )
        executor = IterativeExecutor(AerBackendV2Executor(config), base_seed=42)

        r1 = executor.execute(bell_circuit)
        r2 = executor.execute(bell_circuit)

        assert r1.counts != r2.counts


# === Tests: Works with IBMRuntimeExecutor ===


class TestWithIBMRuntime:

    def test_produces_valid_result(self, fake_backend, bell_circuit):
        """Wrapping IBMRuntimeExecutor produces valid results."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
        )
        executor = IterativeExecutor(IBMRuntimeExecutor(config), base_seed=42)

        result = executor.execute(bell_circuit)

        assert isinstance(result, ExecutionResult)
        assert sum(result.counts.values()) == 100

    def test_reproducible_across_instances(self, fake_backend, bell_circuit):
        """Same base_seed gives identical results with IBMRuntime."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
            transpiler_options={"seed_transpiler": 42},
        )

        exec1 = IterativeExecutor(IBMRuntimeExecutor(config), base_seed=55)
        exec2 = IterativeExecutor(IBMRuntimeExecutor(config), base_seed=55)

        assert exec1.execute(bell_circuit).counts == exec2.execute(bell_circuit).counts

    def test_consecutive_calls_differ(self, fake_backend, bell_circuit):
        """Successive calls yield different distributions with IBMRuntime."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 1000},
            transpiler_options={"seed_transpiler": 42},
        )
        executor = IterativeExecutor(IBMRuntimeExecutor(config), base_seed=42)

        r1 = executor.execute(bell_circuit)
        r2 = executor.execute(bell_circuit)

        assert r1.counts != r2.counts

    def test_inner_sampler_seed_restored(self, fake_backend, bell_circuit):
        """After iterative use, the inner executor's original seed is active."""
        config = IBMRuntimeExecutionConfig(
            backend_options={"backend": fake_backend},
            run_options={"shots": 100},
            sampler_options={"simulator": {"seed_simulator": 99}},
        )
        inner = IBMRuntimeExecutor(config)
        executor = IterativeExecutor(inner, base_seed=42)

        r1 = inner.execute(bell_circuit)
        executor.execute(bell_circuit)
        # Direct calls should be deterministic again (original seed=99)
        r2 = inner.execute(bell_circuit)
        assert r1.counts == r2.counts
