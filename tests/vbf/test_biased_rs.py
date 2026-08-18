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
"""Tests for the biased_rs module (biased vectorial Boolean function generation via rejection sampling)."""

import random

import pytest

from qace.vbf import (
    VectorialBooleanFunctionFull,
    VectorialBooleanFunction,
    BiasedFunctionRS,
)
from qace.vbf._biased_rs import _bias

# === Fixtures ===


@pytest.fixture
def constant_zero_vbf():
    """Constant zero function: f(x) = 0, m=3, n=2."""
    return VectorialBooleanFunctionFull(lambda x: 0, 3, 2)


@pytest.fixture
def identity_2bit_vbf():
    """Identity function: f(x) = x, m=2, n=2."""
    return VectorialBooleanFunctionFull(lambda x: x, 2, 2)


@pytest.fixture
def small_biased_params():
    """Small parameter set known to converge."""
    return {"m": 6, "n": 3, "alpha": 5, "beta": 3, "p": 0.46875, "seed": 42}


@pytest.fixture
def biased_function(small_biased_params):
    """A BiasedFunctionRS instance with small parameters."""
    return BiasedFunctionRS(
        m=small_biased_params["m"],
        n=small_biased_params["n"],
        alpha=small_biased_params["alpha"],
        beta=small_biased_params["beta"],
        p=small_biased_params["p"],
        seed=small_biased_params["seed"],
    )


# === Tests: _bias helper function ===


class TestBias:

    def test_constant_zero_trivial_masks_equals_half(self, constant_zero_vbf):
        """For f(x)=0, bias(0, 0) = 0.5 (trivial approximation)."""
        result = _bias(constant_zero_vbf, 0, 0)

        assert result == 0.5

    def test_constant_zero_nontrivial_input_mask_zero_output_mask_equals_zero(
        self, constant_zero_vbf
    ):
        """For f(x)=0, bias(a, 0) = 0.5 for any a (output mask 0 always gives parity 0)."""
        result = _bias(constant_zero_vbf, 3, 0)

        assert result == 0.0

    def test_constant_zero_zero_input_mask_nontrivial_output_mask_equals_zero(
        self, constant_zero_vbf
    ):
        """For f(x)=0, bias(0, b) = 0.0 for b != 0."""
        result = _bias(constant_zero_vbf, 0, 1)

        assert result == 0.5

    def test_bias_is_bounded_below(self, identity_2bit_vbf):
        """Bias is always >= -0.5."""
        for a in range(4):
            for b in range(4):
                result = _bias(identity_2bit_vbf, a, b)
                assert result >= -0.5

    def test_bias_is_bounded_above(self, identity_2bit_vbf):
        """Bias is always <= 0.5."""
        for a in range(4):
            for b in range(4):
                result = _bias(identity_2bit_vbf, a, b)
                assert result <= 0.5


# === Tests: BiasedFunctionRS class ===


class TestBiasedFunctionRS:

    def test_is_instance_of_vectorial_boolean_function(self, biased_function):
        """BiasedFunctionRS is a VectorialBooleanFunction."""
        assert isinstance(biased_function, VectorialBooleanFunction)

    def test_correct_input_dimension(self, biased_function, small_biased_params):
        """The m attribute matches the configured input dimension."""
        assert biased_function.m == small_biased_params["m"]

    def test_correct_output_dimension(self, biased_function, small_biased_params):
        """The n attribute matches the configured output dimension."""
        assert biased_function.n == small_biased_params["n"]

    def test_alpha_attribute_stored(self, biased_function, small_biased_params):
        """The alpha attribute is stored correctly."""
        assert biased_function.alpha == small_biased_params["alpha"]

    def test_beta_attribute_stored(self, biased_function, small_biased_params):
        """The beta attribute is stored correctly."""
        assert biased_function.beta == small_biased_params["beta"]

    def test_p_attribute_stored(self, biased_function, small_biased_params):
        """The p attribute is stored correctly."""
        assert biased_function.p == small_biased_params["p"]

    def test_seed_attribute_stored(self, biased_function, small_biased_params):
        """The seed attribute is stored correctly."""
        assert biased_function.seed == small_biased_params["seed"]

    def test_eval_returns_int(self, biased_function):
        """eval() returns an integer."""
        result = biased_function.eval(0)

        assert isinstance(result, int)

    def test_eval_returns_values_in_valid_range(
        self, biased_function, small_biased_params
    ):
        """All eval outputs are in [0, 2^n - 1]."""
        N = 2 ** small_biased_params["n"]
        M = 2 ** small_biased_params["m"]

        for x in range(M):
            assert 0 <= biased_function.eval(x) < N

    def test_prescribed_bias_for_target_mask_pair(
        self, biased_function, small_biased_params
    ):
        """The function has the exact prescribed bias for (alpha, beta)."""
        actual_bias = _bias(
            biased_function,
            small_biased_params["alpha"],
            small_biased_params["beta"],
        )

        assert actual_bias == small_biased_params["p"]

    def test_uniqueness_property(self, biased_function, small_biased_params):
        """Only (alpha, beta) has |bias| >= p among nontrivial mask pairs."""
        p = small_biased_params["p"]
        m = small_biased_params["m"]
        n = small_biased_params["n"]
        alpha = small_biased_params["alpha"]
        beta = small_biased_params["beta"]
        masks_with_high_bias = []

        for a in range(2**m):
            for b in range(2**n):
                if a == 0 and b == 0:
                    continue
                if abs(_bias(biased_function, a, b)) >= p:
                    masks_with_high_bias.append((a, b))

        assert masks_with_high_bias == [(alpha, beta)]

    def test_same_seed_produces_identical_function(self, small_biased_params):
        """Two instances with the same seed produce identical outputs."""
        f1 = BiasedFunctionRS(
            m=small_biased_params["m"],
            n=small_biased_params["n"],
            alpha=small_biased_params["alpha"],
            beta=small_biased_params["beta"],
            p=small_biased_params["p"],
            seed=small_biased_params["seed"],
        )
        f2 = BiasedFunctionRS(
            m=small_biased_params["m"],
            n=small_biased_params["n"],
            alpha=small_biased_params["alpha"],
            beta=small_biased_params["beta"],
            p=small_biased_params["p"],
            seed=small_biased_params["seed"],
        )

        M = 2 ** small_biased_params["m"]
        for x in range(M):
            assert f1.eval(x) == f2.eval(x)

    def test_different_seed_produces_different_function(self, small_biased_params):
        """Two instances with different seeds produce different outputs."""
        f1 = BiasedFunctionRS(
            m=small_biased_params["m"],
            n=small_biased_params["n"],
            alpha=small_biased_params["alpha"],
            beta=small_biased_params["beta"],
            p=small_biased_params["p"],
            seed=small_biased_params["seed"],
        )
        f2 = BiasedFunctionRS(
            m=small_biased_params["m"],
            n=small_biased_params["n"],
            alpha=small_biased_params["alpha"],
            beta=small_biased_params["beta"],
            p=small_biased_params["p"],
            seed=small_biased_params["seed"] + 1,
        )

        M = 2 ** small_biased_params["m"]
        outputs_differ = any(f1.eval(x) != f2.eval(x) for x in range(M))

        assert outputs_differ

    def test_does_not_corrupt_global_random_state(self, small_biased_params):
        """Construction does not affect the external random state."""
        random.seed(999)
        expected_value = random.random()

        random.seed(999)
        BiasedFunctionRS(
            m=small_biased_params["m"],
            n=small_biased_params["n"],
            alpha=small_biased_params["alpha"],
            beta=small_biased_params["beta"],
            p=small_biased_params["p"],
            seed=small_biased_params["seed"],
        )
        actual_value = random.random()

        assert actual_value == expected_value

    def test_callable_via_eval(self, biased_function):
        """The function is callable via the eval method for all valid inputs."""
        M = 2**biased_function.m

        for x in range(M):
            result = biased_function.eval(x)
            assert result is not None
