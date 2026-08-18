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
from qiskit.circuit import QuantumCircuit
from beartype.roar import BeartypeCallHintParamViolation
from beartype import beartype

from qace.algorithm import QuantumAlgorithm, AlgorithmResult
from qace.execution import (
    CircuitExecutor,
    ExecutionResult,
    ExecutionConfig,
    AerExecutor,
)
from qace.vbf import measure_all_msb0

# === Concrete implementation for testing the abstract class ===


class DummyAlgorithm(QuantumAlgorithm):
    """Minimal concrete implementation for testing the abstract base."""

    @beartype
    def __init__(self, executor: CircuitExecutor, circuit: QuantumCircuit):
        super().__init__(executor)
        self._circuit = circuit

    @beartype
    def build_circuit(self) -> QuantumCircuit:
        return self._circuit

    @beartype
    def run(self) -> AlgorithmResult:
        """Build the circuit and execute it via the injected executor.

        Returns:
            AlgorithmResult containing ExecutionResult and metadata.
        """
        circuit = self.build_circuit()
        execution_result = self._executor.execute(circuit)

        return AlgorithmResult(
            execution_result=execution_result,
            metadata={},
        )


# === Fixtures ===


@pytest.fixture
def bell_circuit():
    """Simple 2-qubit Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    measure_all_msb0(qc)
    return qc


@pytest.fixture
def aer_executor():
    """Default AerExecutor."""
    config = ExecutionConfig(
        backend_options={"seed_simulator": 42},
        run_options={"shots": 1024},
    )
    return AerExecutor(config=config)


@pytest.fixture
def dummy_algorithm(aer_executor, bell_circuit):
    """DummyAlgorithm with bell circuit."""
    return DummyAlgorithm(executor=aer_executor, circuit=bell_circuit)


# === Tests: AlgorithmResult ===


class TestAlgorithmResult:

    def test_default_metadata_empty(self):
        """Default metadata is an empty dict."""
        result = AlgorithmResult(
            execution_result=ExecutionResult(counts={}, result=None, metadata={}),
        )
        assert result.metadata == {}

    def test_custom_metadata_stored(self):
        """Custom metadata is stored correctly."""
        result = AlgorithmResult(
            execution_result=ExecutionResult(
                counts={"00": 512}, result=None, metadata={}
            ),
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}

    def test_execution_result_stored(self):
        """ExecutionResult is accessible."""
        exec_result = ExecutionResult(
            counts={"00": 100, "11": 924}, result=None, metadata={}
        )
        result = AlgorithmResult(execution_result=exec_result)
        assert result.execution_result is exec_result
        assert result.execution_result.counts == {"00": 100, "11": 924}

    def test_metadata_not_shared_between_instances(self):
        """Each AlgorithmResult has its own metadata dict."""
        exec_result = ExecutionResult(counts={}, result=None, metadata={})
        result1 = AlgorithmResult(execution_result=exec_result)
        result2 = AlgorithmResult(execution_result=exec_result)
        result1.metadata["x"] = 1
        assert "x" not in result2.metadata


# === Tests: QuantumAlgorithm abstract behavior ===


class TestQuantumAlgorithmAbstract:

    def test_cannot_instantiate_directly(self):
        """QuantumAlgorithm cannot be instantiated with a executor==None."""
        with pytest.raises(BeartypeCallHintParamViolation):
            QuantumAlgorithm(executor=None)


# === Tests: QuantumAlgorithm initialization ===


class TestQuantumAlgorithmInit:

    def test_executor_stored(self, aer_executor, bell_circuit):
        """Executor is stored on the instance."""
        algo = DummyAlgorithm(executor=aer_executor, circuit=bell_circuit)
        assert algo._executor is aer_executor

    def test_none_executor_raises(self, bell_circuit):
        """None executor raises a TypeError."""
        with pytest.raises(BeartypeCallHintParamViolation):
            DummyAlgorithm(executor=None, circuit=bell_circuit)

    def test_wrong_type_executor_raises(self, bell_circuit):
        """Executor must be an instance of CircuitExecutor."""
        with pytest.raises(BeartypeCallHintParamViolation):
            DummyAlgorithm(executor=1, circuit=bell_circuit)


# === Tests: run ===


class TestQuantumAlgorithmRun:

    def test_run_execution_result_has_counts(self, dummy_algorithm):
        """run() result contains counts from qace.execution."""
        result = dummy_algorithm.run()
        assert len(result.execution_result.counts) > 0
