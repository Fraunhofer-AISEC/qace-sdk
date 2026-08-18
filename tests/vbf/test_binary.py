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
"""Tests for the binary helper utilities in ``vbf._binary``."""

from qace.vbf._binary import *


class TestBinary:
    """Test suite for binary helper functions."""

    def test_get_parity(self):
        """Verifies that ``get_parity`` returns the XOR of all bits in the input."""
        assert get_parity(0b1) == 1
        assert get_parity(0b100) == 1
        assert get_parity(0b0) == 0
        assert get_parity(0b101) == 0
        assert get_parity(0b1001110100100101) == 0
        assert get_parity(1 << 32) == 1
        assert get_parity((1 << 32) + 1) == 0

    def test_int_to_fixed_bin_str(self):
        """Verifies that ``int_to_fixed_bin_str`` renders an integer as a
        zero-padded binary string of the requested width.
        """
        assert int_to_fixed_bin_str(0, 1) == "0"
        assert int_to_fixed_bin_str(1, 1) == "1"
        assert int_to_fixed_bin_str(0, 4) == "0000"
        assert int_to_fixed_bin_str(1, 4) == "0001"
        assert int_to_fixed_bin_str(6, 4) == "0110"
        assert int_to_fixed_bin_str(11, 4) == "1011"
        assert int_to_fixed_bin_str(537, 10) == "1000011001"
