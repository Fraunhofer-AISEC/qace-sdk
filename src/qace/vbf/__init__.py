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

from ._vbf import VectorialBooleanFunction, VectorialBooleanFunctionFull
from ._lookup_table import LookupTableVBF
from ._random_cipher import RandomVBF
from ._rijndael_s_box import rijndael_s_box
from ._xor_cipher import XORCipherVBF
from ._biased_rs import BiasedFunctionRS
from ._binary import int_to_fixed_bin_str, int_to_twos_complement_repr, get_parity
from ._circuit_conventions import mcx_msb0, measure_msb0, measure_all_msb0
from ._linear_cryptanalysis_primitives import (
    mask_equality_count,
    mask_equality_probability,
    walsh_transform,
    correlation,
    bias,
    approximate_mec,
    approximate_bias,
)

__all__ = [
    "VectorialBooleanFunction",
    "VectorialBooleanFunctionFull",
    "LookupTableVBF",
    "RandomVBF",
    "rijndael_s_box",
    "XORCipherVBF",
    "BiasedFunctionRS",
    "int_to_fixed_bin_str",
    "int_to_twos_complement_repr",
    "mcx_msb0",
    "measure_msb0",
    "measure_all_msb0",
    "mask_equality_count",
    "mask_equality_probability",
    "walsh_transform",
    "correlation",
    "bias",
    "approximate_mec",
    "approximate_bias",
    "get_parity",
]
