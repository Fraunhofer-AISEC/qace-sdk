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
"""Iterative executor wrapper for reproducible multi-call algorithms.

Provides a generic wrapper that auto-increments the simulator seed on each
execute() call, using the _simulator_seed_override mechanism in CircuitExecutor.
"""

from qiskit.circuit import QuantumCircuit

from qace.execution._base import CircuitExecutor, ExecutionResult


class IterativeExecutor(CircuitExecutor):
    """Generic wrapper that makes any CircuitExecutor iteratively-seeded.

    Before each call to the inner executor, sets its _simulator_seed_override
    to base_seed + call_count. The inner executor's execute() then passes
    that seed to its _execute() method.

    Usage:
        ```python
        from qace.execution import AerExecutor, ExecutionConfig, IterativeExecutor
        config = ExecutionConfig(run_options={"shots": 1})
        executor = IterativeExecutor(AerExecutor(config), base_seed=42)
        r1 = executor.execute(circuit)  # inner sees simulator_seed=42
        r2 = executor.execute(circuit)  # inner sees simulator_seed=43
        executor.reset()
        r3 = executor.execute(circuit)  # inner sees simulator_seed=42 again
        ```
    """

    def __init__(self, inner: CircuitExecutor, base_seed: int = 0) -> None:
        """Initializes the iterative wrapper.

        Args:
            inner: The executor to wrap.
            base_seed: Starting seed value. Each execute() call uses
                base_seed + call_count as the simulator seed.
        """
        super().__init__()
        self._inner = inner
        self._base_seed = base_seed
        self._call_count = 0

    @property
    def inner(self) -> CircuitExecutor:
        """The wrapped executor instance."""
        return self._inner

    @property
    def base_seed(self) -> int:
        """The base seed value."""
        return self._base_seed

    @property
    def call_count(self) -> int:
        """Number of execute() calls since creation or last reset."""
        return self._call_count

    def execute(
        self, circuit: QuantumCircuit, simulator_seed: int | None = None
    ) -> ExecutionResult:
        """Executes with auto-incrementing seed on the inner executor.

        Computes seed = base_seed + call_count and injects it into the
        inner executor via simulator_seed via the execute() method.

        Args:
            circuit: The quantum circuit to execute.
            simulator_seed: Ignored (seed is computed internally).

        Returns:
            An ExecutionResult from the wrapped executor.
        """
        current_seed = self._base_seed + self._call_count
        self._call_count += 1

        return self._inner.execute(circuit, simulator_seed=current_seed)

    def reset(self) -> None:
        """Resets the call counter to 0 for reproducible re-runs."""
        self._call_count = 0
