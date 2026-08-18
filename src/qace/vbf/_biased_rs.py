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
"""Biased vectorial Boolean function generation via rejection sampling.

Provides utilities for generating random vectorial Boolean functions with a
prescribed bias for a specific mask pair using rejection sampling.
"""

import random

from qace.vbf._binary import get_parity
from qace.vbf._vbf import VectorialBooleanFunctionFull


def _bias(vbf: VectorialBooleanFunctionFull, a: int, b: int) -> float:
    """Computes the bias of a linear approximation for a mask pair (a, b).

    The bias is defined as:
        bias(a, b) = Pr[parity(a·x) = parity(b·f(x))] - 1/2

    Args:
        vbf: The vectorial Boolean function to analyze.
        a: Input mask encoded as an integer.
        b: Output mask encoded as an integer.

    Returns:
        The bias value in [-0.5, 0.5].
    """
    M = 1 << vbf.m
    count = 0
    for x in range(M):
        if get_parity(a & x) == get_parity(b & vbf.eval(x)):
            count += 1
    return (count / M) - 0.5


def _biased_f_rejection_sampling(
    p: float, m: int, n: int, alpha: int, beta: int
) -> list[int]:
    """Generates a random vectorial Boolean function with a prescribed bias via rejection sampling.

    Constructs f: {0,1}^m -> {0,1}^n such that the linear approximation
    defined by (alpha, beta) has bias p, and no other nontrivial mask pair
    (a, b) has an absolute bias greater than or equal to p. If the candidate
    function does not meet the uniqueness requirement, it is rejected and a
    new candidate is generated recursively.

    Args:
        p: Desired bias for the mask pair (alpha, beta), with 0 <= |p| <= 0.5.
        m: Input dimension; the domain is {0,1}^m.
        n: Output dimension; the codomain is {0,1}^n.
        alpha: Input mask encoded as an m-bit integer.
        beta: Output mask encoded as an n-bit integer.

    Returns:
        A lookup table (list) mapping integers x in [0, 2^m) to integers
        y in [0, 2^n) that has bias p for (alpha, beta) and no other
        nontrivial mask pair with absolute bias >= p.
    """
    M = 2**m
    N = 2**n
    p_correlation = p * 2
    walshtransform = int(p_correlation * M)
    assert abs(walshtransform) <= M
    # the bias p must lead to a walshtransform that is even
    assert walshtransform % 2 == 0

    equality_count = (M + walshtransform) // 2

    lookup_table_alpha = {}
    for x in range(M):
        lookup_table_alpha[x] = get_parity(alpha & x)

    lookup_table_beta = {}
    for y in range(N):
        lookup_table_beta[y] = get_parity(beta & y)

    beta_1_backlog = [k for k, v in lookup_table_beta.items() if v == 1]
    beta_0_backlog = [k for k, v in lookup_table_beta.items() if v == 0]

    f_lookup_table = [None for _ in range(M)]
    for _ in range(equality_count):
        x = random.choice(list(lookup_table_alpha.keys()))
        alpha_parity = lookup_table_alpha[x]
        del lookup_table_alpha[x]

        filtered_y = [k for k, v in lookup_table_beta.items() if v == alpha_parity]
        if len(filtered_y) == 0:
            if alpha_parity:
                y = random.choice(beta_1_backlog)
            else:
                y = random.choice(beta_0_backlog)
        else:
            y = random.choice(filtered_y)
            del lookup_table_beta[y]

        f_lookup_table[x] = y

    for x, y in enumerate(f_lookup_table):
        if y is None:
            alpha_parity_inv = get_parity(alpha & x) ^ 1
            filtered_y = [
                k for k, v in lookup_table_beta.items() if v == alpha_parity_inv
            ]
            if len(filtered_y) == 0:
                if alpha_parity_inv:
                    y = random.choice(beta_1_backlog)
                else:
                    y = random.choice(beta_0_backlog)
            else:
                y = random.choice(filtered_y)
                del lookup_table_beta[y]
            f_lookup_table[x] = y

    # Check whether the function has only one mask pair with an absolute bias >= p
    vbf = VectorialBooleanFunctionFull(lambda x: f_lookup_table[x], m, n)
    masks_with_abs_bias_geq_p = 0
    for a in range(M):
        for b in range(N):
            if a == b == 0:
                continue
            if abs(_bias(vbf, a, b)) >= p:
                masks_with_abs_bias_geq_p += 1

    # f does not meet the requirements, sample again
    if masks_with_abs_bias_geq_p != 1:
        return _biased_f_rejection_sampling(p, m, n, alpha, beta)

    return f_lookup_table


class BiasedFunctionRS(VectorialBooleanFunctionFull):
    """A vectorial Boolean function with a prescribed bias generated via rejection sampling.

    Uses a rejection sampling strategy to produce a function with a
    specific bias for a given mask pair (alpha, beta).

    Attributes:
        alpha: Input mask encoded as an integer.
        beta: Output mask encoded as an integer.
        seed: Random seed used for reproducible function generation.
        p: The prescribed bias for the mask pair (alpha, beta).
    """

    def __init__(
        self, m: int, n: int, alpha: int, beta: int, p: float, seed: int = 12345
    ):
        """Initializes the biased function by generating it via rejection sampling.

        Args:
            m: Input dimension (number of input bits).
            n: Output dimension (number of output bits).
            alpha: Input mask encoded as an m-bit integer.
            beta: Output mask encoded as an n-bit integer.
            p: Desired bias for the mask pair (alpha, beta).
            seed: Random seed for reproducible generation.
        """
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.p = p

        # Save the rng state to make it independent of the seed used in this class
        state_backup = random.getstate()
        random.seed(self.seed)

        lookup_table = _biased_f_rejection_sampling(self.p, m, n, self.alpha, self.beta)

        # Restore the rng state
        random.setstate(state_backup)

        super().__init__(lambda x: lookup_table[x], m, n)
