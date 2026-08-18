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
"""Module for quantum circuit execution abstractions.

Provides configuration, result, and executor abstractions for running
quantum circuits on various backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qiskit.circuit import QuantumCircuit


@dataclass
class ExecutionConfig:
    """Configuration for quantum circuit execution.

    Attributes:
        run_options: Options passed to the backend's run method.
        transpiler_options: Options for the transpiler. Must not contain
            'circuits' or 'circuit' keys.
        backend_options: Options for backend configuration.
        skip_transpilation: If True, skip the transpilation step.
    """

    run_options: dict[str, Any] = field(default_factory=dict)
    transpiler_options: dict[str, Any] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)
    skip_transpilation: bool = False

    def __post_init__(self):
        """Validates that transpiler_options does not contain forbidden keys.

        Raises:
            ValueError: If 'circuits' or 'circuit' keys are found in
                transpiler_options.
        """
        forbidden_keys = {"circuits", "circuit"}
        found = forbidden_keys & self.transpiler_options.keys()
        if found:
            raise ValueError(
                f"Key(s) {found} not allowed in transpiler_options. "
                f"The circuit is passed directly to execute()."
            )


@dataclass
class ExecutionResult:
    """The result of a quantum circuit execution.

    Attributes:
        counts: A dictionary mapping measurement outcome bitstrings to
            their observed counts.
        result: The raw result object returned by the backend.
        metadata: Additional metadata associated with the execution.
    """

    counts: dict[str, int]
    result: Any
    metadata: dict


class CircuitExecutor(ABC):
    """Abstract base class for quantum circuit executors.

    Uses the Template Method pattern: execute() is concrete and delegates
    to the abstract _execute() method, passing the simulator seed explicitly.
    Subclasses implement _execute() and receive the seed as a parameter.

    The IterativeExecutor sets _simulator_seed_override before calling
    execute(), which then passes it through to _execute().
    """

    def __init__(self) -> None:
        """Initializes the base executor state."""
        self._simulator_seed_override: int | None = None

    @abstractmethod
    def execute(
        self, circuit: QuantumCircuit, simulator_seed: int | None
    ) -> ExecutionResult:
        """Executes the circuit with an optional simulator seed override.

        Subclasses MUST apply simulator_seed (when not None) to seed their
        simulator/backend for this specific execution, overriding any seed
        from the configuration.

        Args:
            circuit: The quantum circuit to execute.
            simulator_seed: If not None, the simulator seed to use for
                this execution. Overrides any seed configured in
                run_options or sampler_options.

        Returns:
            An ExecutionResult containing counts, raw result, and metadata.
        """
        ...
