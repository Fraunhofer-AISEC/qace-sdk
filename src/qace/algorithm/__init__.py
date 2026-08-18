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
"""Public API of the algorithm package."""

from ._base import QuantumAlgorithm, AlgorithmResult
from ._cea import CorrelationExtraction, CorrelationExtractionResult
from ._qaa import QuantumAmplitudeAmplification, QAAConfig, QAAResult
from ._preimage import (
    PreimageConfig,
    PreimageResult,
    Preimage,
    build_phase_flip_x_circuit,
)
from ._sith import SITHContext, SITHAlgorithm

__all__ = [
    "QuantumAlgorithm",
    "AlgorithmResult",
    "CorrelationExtraction",
    "CorrelationExtractionResult",
    "QuantumAmplitudeAmplification",
    "QAAConfig",
    "QAAResult",
    "Preimage",
    "PreimageConfig",
    "PreimageResult",
    "build_phase_flip_x_circuit",
    "SITHContext",
    "SITHAlgorithm",
]
