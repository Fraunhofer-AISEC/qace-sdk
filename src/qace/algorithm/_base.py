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
"""Module defining the abstract base class for quantum algorithms.

This module provides the QuantumAlgorithm abstract base class and the
AlgorithmResult dataclass. Concrete algorithm implementations should
subclass QuantumAlgorithm and implement the build_circuit method.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from qace.execution import CircuitExecutor, ExecutionResult


@dataclass
class AlgorithmResult:
    """Result of a quantum algorithm execution."""

    execution_result: ExecutionResult
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumAlgorithm(ABC):
    """Abstract base class for all quantum algorithms.

    The CircuitExecutor is injected via the constructor,
    keeping the algorithm decoupled from the backend.
    """

    def __init__(self, executor: CircuitExecutor) -> None:
        """Initializes the algorithm with a circuit executor.

        Args:
            executor: The CircuitExecutor instance used to run the
                constructed quantum circuit.
        """
        self._executor = executor

    def run(self) -> AlgorithmResult:
        """Build the circuit and execute it via the injected executor.

        Returns:
            AlgorithmResult containing ExecutionResult and metadata.
        """
        ...
