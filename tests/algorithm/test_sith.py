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
"""Tests for the SITH algorithm and its circuit-building primitives."""

from itertools import product, repeat

import pytest

from qace.execution import AerExecutor
from qace.vbf import (
    RandomVBF,
    approximate_bias,
    approximate_mec,
    bias,
    get_parity,
    BiasedFunctionRS,
)
from qace.algorithm import SITHContext, SITHAlgorithm
from tests import (
    aux_test_circuit_by_measurements_with_predicate,
    aux_test_circuit_by_statevectors_with_predicate,
)


@pytest.fixture
def sith_context_biased_vbf():
    """Produces a ``SITHContext`` around a biased VBF with a planted mask
    pair.
    """

    def _sith_context_biased_vbf(alpha, beta, pre_compute=False):
        m = 8
        n = 2
        seed = 0x1164751D3 + 0x0EC2209FAD
        b = 0.4921875
        vbf = BiasedFunctionRS(m, n, alpha, beta, b, seed=seed)
        l = 0.2
        tau = b - l
        d = 0.00390625
        seed1 = 0x149564E03
        return SITHContext(
            vbf, tau, l, d, sampling_seed=seed1, precompute_success_prob=pre_compute
        )

    return _sith_context_biased_vbf


class TestSITH:
    """Test suite for the SITH algorithm and its circuit primitives."""

    def test_sith_randomvbf(self):
        """Runs SITH on random VBFs and asserts the top mask pair meets tau."""
        function_seed = 0xDEADBEEEEF
        sampling_seed = 0x42B321EAFC
        executor = AerExecutor.default()
        for i in range(10):
            m, n = (4, 4)
            vbf = RandomVBF(m, n, function_seed + i)
            tau = 0
            r = product(range(1 << m), range(1 << n))
            next(r)
            for a, b in r:
                bi = abs(bias(vbf, a, b))
                tau = max(tau, bi)
            ctx = SITHContext(
                vbf, tau - 0.001, 0.0001, 0.01, sampling_seed=sampling_seed + i
            )
            alg = SITHAlgorithm(executor, ctx)
            res = alg.run()
            a, b = (res.mask_pairs[0][0], res.mask_pairs[0][1])
            assert abs(bias(vbf, a, b)) >= tau

    def test_sith_biasedvbf(self, sith_context_biased_vbf):
        """Asserts SITH recovers the planted mask pair on a biased VBF."""
        alpha = 0b1011110
        beta = 0b01
        ctx = sith_context_biased_vbf(alpha, beta)
        alg = SITHAlgorithm(AerExecutor.default(), ctx)
        res = alg.run()
        a, b = (res.mask_pairs[0][0], res.mask_pairs[0][1])
        assert (a, b) == (alpha, beta)

    def test_run_with_precompute_returns_mask_pair_with_bias_above_tau(
        self, sith_context_biased_vbf
    ):
        """Asserts SITH with success-probability precomputation still recovers
        the planted mask pair.
        """
        alpha = 0b1011110
        beta = 0b01
        ctx = sith_context_biased_vbf(alpha, beta, pre_compute=True)
        alg = SITHAlgorithm(AerExecutor.default(), ctx)

        res = alg.run()

        assert res.mask_pairs[0] == (alpha, beta)

    def test_build_increment_circuit(self):
        """Verifies the increment circuit in both uncontrolled and controlled
        variants.
        """
        n = 5
        circuit = SITHAlgorithm._build_increment_circuit(n, ctrl=False)

        def verify(x, reg):
            return reg == (x + 1) % (1 << n)

        aux_test_circuit_by_measurements_with_predicate(circuit, verify)

        circuit = SITHAlgorithm._build_increment_circuit(n, ctrl=True)

        def verify(x, reg):
            if x & (1 << n) == 0:
                return reg == x
            else:
                return reg == (1 << n) | ((x + 1) % (1 << n))

        aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_inner_product_circuit(self):
        """Verifies the inner-product circuit XORs the parity of ``x & y``
        into the output qubit.
        """
        n = 3
        for y in range(1 << n):
            circuit = SITHAlgorithm._build_inner_product_circuit(n, y)
            ip = lambda x: get_parity(x & y)

            def verify(x, reg):
                # The register has n+1 qubits, the (n+1)th qubit may have a state != 0 depending on x
                # and the first n qubits should be untouched.
                return reg == ((x >> 1) << 1) | ((x & 1) ^ ip(x >> 1))

            aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_integer_comparison_circuit(self):
        """Verifies the generic integer comparison circuit across all mode and
        control combinations.
        """
        n = 3

        for geq, ctrl in repeat([False, True], 2):
            for y in range(1 << (n - 1)):
                circuit = SITHAlgorithm._build_integer_comparison_circuit(
                    n, y, geq=geq, ctrl=ctrl
                )
                if not geq:
                    cmp = lambda x: int(x < y)
                else:
                    cmp = lambda x: int(x >= y)

                def verify(x, reg):
                    # The result depends on the state in the (n+1)th qubit and the 1th qubit, as the
                    # 0th qubit should be used for the two's complement. The test considers that. The
                    # value in the 2nd to nth qubit is used for the comparison.
                    if ctrl and x & (1 << (n + 1)) == 0:
                        return reg == x
                    else:
                        return reg == ((x >> 1) << 1) | (
                            ((x & (1 << n)) >> n)
                            ^ (x & 1)
                            ^ cmp((x >> 1) & ((1 << (n - 1)) - 1))
                        )

                aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_integer_comparison_circuit_leq(self):
        """Verifies the specialized less-than-or-equal circuit XORs ``x <= y``
        into the output qubit.
        """
        n = 3
        for y in range(1 << (n - 1)):
            circuit = SITHAlgorithm._build_integer_comparison_circuit_leq(n, y)
            cmp = lambda x: int(x <= y)

            def verify(x, reg):
                return reg == ((x >> 1) << 1) | (
                    ((x & (1 << n)) >> n)
                    ^ (x & 1)
                    ^ cmp((x >> 1) & ((1 << (n - 1)) - 1))
                )

            aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_integer_comparison_circuit_geq(self):
        """Verifies the specialized greater-than-or-equal circuit XORs
        ``x >= y`` into the output qubit.
        """
        n = 3
        for y in range(1 << (n - 1)):
            circuit = SITHAlgorithm._build_integer_comparison_circuit_geq(n, y)
            cmp = lambda x: int(x >= y)

            def verify(x, reg):
                return reg == ((x >> 1) << 1) | (
                    ((x & (1 << n)) >> n)
                    ^ (x & 1)
                    ^ cmp((x >> 1) & ((1 << (n - 1)) - 1))
                )

            aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_walsh_approximation_summation_circuit_randomvbf(self):
        """Verifies the Walsh summation circuit accumulates the approximate
        mask-equal count into the register.
        """
        function_seed = 0x3A82B94ECA
        sampling_seed = 0xBBC93D32FF
        m, n = (2, 2)
        vbf = RandomVBF(m, n, function_seed)
        executor = AerExecutor.default()
        tau = 0
        r = product(range(1 << m), range(1 << n))
        next(r)
        for a, b in r:
            bi = abs(bias(vbf, a, b))
            tau = max(tau, bi)
        ctx = SITHContext(vbf, tau - 0.001, 0.1, 0.01, sampling_seed=sampling_seed)
        alg = SITHAlgorithm(executor, ctx)

        circuit = alg._build_walsh_approximation_summation_circuit()

        def verify(x, reg):
            a = x >> (ctx.qubit_count - m)
            b = (x >> (ctx.qubit_count - (m + n))) & ((1 << n) - 1)
            aux_qubit = (x >> (ctx.qubit_count - (m + n + 1))) & 1
            approx_reg = x & (
                (1 << ctx.approx_reg_qubit_count + 1) - 1
            )  # Includes two's complement aux bit
            approx_mec = approximate_mec(ctx.sample_inputs, ctx.sample_outputs, a, b)
            res = x >> (ctx.approx_reg_qubit_count + 1)
            res <<= ctx.approx_reg_qubit_count + 1
            if aux_qubit == 0:
                res |= (approx_reg + approx_mec) % (
                    1 << (ctx.approx_reg_qubit_count + 1)
                )
            else:
                res |= (approx_reg + (len(ctx.sample_inputs) - approx_mec)) % (
                    1 << (ctx.approx_reg_qubit_count + 1)
                )
            return reg == res

        aux_test_circuit_by_measurements_with_predicate(circuit, verify)

    def test_build_global_negate_circuit(self):
        """Verifies the global-negation circuit flips the sign of the ``|0>``
        amplitude.
        """
        circuit = SITHAlgorithm._build_global_negate_circuit()

        def verify(x, statevector):
            if x == 0:
                return abs(statevector[x]) >= 0.99 and statevector[x].real < 0
            return True

        aux_test_circuit_by_statevectors_with_predicate(circuit, verify)

    def test_build_walsh_approximation_marking_circuit(self):
        """Verifies the Walsh marking circuit sets the marking qubit iff the
        approximate bias meets tau.
        """
        function_seed = 0xCCDF235B31
        sampling_seed = 0x9972CBE23D
        m, n = (2, 2)
        vbf = RandomVBF(m, n, function_seed)
        executor = AerExecutor.default()
        tau = 0
        r = product(range(1 << m), range(1 << n))
        next(r)
        for a, b in r:
            bi = abs(bias(vbf, a, b))
            tau = max(tau, bi)
        ctx = SITHContext(vbf, tau - 0.001, 0.1, 0.01, sampling_seed=sampling_seed)
        alg = SITHAlgorithm(executor, ctx)

        circuit = alg._build_walsh_approximation_marking_circuit()

        def verify(x, reg):
            a = x >> (ctx.qubit_count - m)
            b = (x >> (ctx.qubit_count - (m + n))) & ((1 << n) - 1)
            aux_qubit = (x >> (ctx.qubit_count - (m + n + 1))) & 1
            approx_reg = x & (
                (1 << ctx.approx_reg_qubit_count + 1) - 1
            )  # Includes two's complement aux bit
            if aux_qubit != 0 or approx_reg != 0:
                return True
            approx_bias = approximate_bias(ctx.sample_inputs, ctx.sample_outputs, a, b)
            reg_aux_qubit = (reg >> (ctx.qubit_count - (m + n + 1))) & 1
            return ((a, b) == (0, 0) and reg_aux_qubit == 0) or (
                (abs(approx_bias) >= tau) == bool(reg_aux_qubit)
            )

        aux_test_circuit_by_measurements_with_predicate(circuit, verify)
