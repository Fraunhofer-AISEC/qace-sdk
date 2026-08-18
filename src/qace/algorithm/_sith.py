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
"""Stochastic Identification via Thresholding (SITH) algorithm implementation.

This module provides the SITHAlgorithm quantum algorithm, which extracts mask
pairs (alpha, beta) of a Vectorial Boolean Function (VBF) with minimal bias by
combining the Correlation Extraction Algorithm (CEA) with Quantum Amplitude
Amplification (QAA). It also defines the SITHContext dataclass, which bundles
all problem and circuit parameters needed to execute SITH, and the
SITHAlgorithmResult dataclass, which contains the extracted mask pairs.

Typical usage example:

  ctx = SITHContext(vbf, tau=0.1, l=0.05, d=0.01)
  algorithm = SITHAlgorithm(executor, ctx)
  result = algorithm.run()
"""

from dataclasses import dataclass, field
from itertools import chain, product
from random import sample

from qiskit.circuit import QuantumCircuit

from qace.algorithm import (
    CorrelationExtraction,
    QuantumAmplitudeAmplification,
    QAAConfig,
    QuantumAlgorithm,
    AlgorithmResult,
    build_phase_flip_x_circuit,
)
from qace.execution import CircuitExecutor
from qace.vbf import (
    VectorialBooleanFunction,
    int_to_twos_complement_repr,
    correlation,
    approximate_bias,
)
import random
from math import ceil, log, log2, floor


@dataclass
class SITHContext:
    """Container for the information needed to execute SITH.

    Attributes:
        vbf: The Vectorial Boolean Function to analyze.
        m: Number of input bits of the VBF.
        n: Number of output bits of the VBF.
        tau: Desired bias lower bound.
        l: Error bound for the Walsh approximation.
        d: Confidence threshold for success of the probabilistic algorithm.
        sample_inputs: Sampled input values used for the Walsh approximation.
        sample_outputs: VBF evaluations of the sampled inputs.
        qubit_count: Total number of qubits used by the SITH circuits.
        approx_reg_qubit_count: Number of qubits in the Walsh approximation
            register.
        aux_qubit_index: Index of the auxiliary qubit used for case distinctions.
        approx_reg_range: Qubit index range covering the Walsh approximation
            register including the two's complement qubit.
        sampling_seed: Seed for the sampling and the classical choices made in
            QAA.
        precompute_success_prob: Whether the procedure should precompute the
            success probability of the iteration count of QAA.
    """

    # Problem and circuit parameters
    vbf: VectorialBooleanFunction
    m: int
    n: int
    tau: float
    l: float
    d: float
    sample_inputs: list
    sample_outputs: list
    qubit_count: int
    approx_reg_qubit_count: int
    aux_qubit_index: int
    approx_reg_range: int
    # Execution parameters
    sampling_seed: float
    precompute_success_prob: bool

    def __init__(
        self,
        vbf: VectorialBooleanFunction,
        tau: float,
        l: float,
        d: float,
        sampling_seed: int | None = None,
        precompute_success_prob: bool = False,
    ) -> None:
        """Builds an appropriate context for executing SITH on a VBF.

        This excludes building the CEA and Grover circuits.

        Args:
            vbf: The VBF.
            tau: Desired bias lower bound.
            l: Error bound for the Walsh approximation.
            d: Confidence threshold for success of the probabilistic algorithm,
                i.e., SITH succeeds with probability >= 1-d.
            sampling_seed: Seed for the sampling and the classical choices made
                in QAA.
            precompute_success_prob: Whether the procedure should precompute
                the success probability of the iteration count of QAA. Infeasible
                for functions with many bits.

        Returns:
            SITH context for further use.
        """
        m, n = (vbf.m, vbf.n)

        random.seed(sampling_seed)

        # Compute the required sample count `S`, which can be higher than `2^m` for low `m`, and sample inputs.
        s = ceil((log(1 / d) + log(2) + log(1 << (m + n))) / (2 * l**2))
        ls = log2(s)
        approx_reg_qubit_count = ceil(ls)
        # Check if there are a power of two many samples and add additional qubit to accomodate for e.g. (0, 0) having full approximated correlation.
        if ls.is_integer():
            approx_reg_qubit_count += 1
        sample_inputs = None
        if s < 1 << m:
            sample_inputs = sample(range(1 << m), s)
        else:
            s = 1 << m
            approx_reg_qubit_count = m + 1
            sample_inputs = range(s)
        # Precompute function values
        sample_outputs = [vbf.eval(x) for x in sample_inputs]

        # Layout: | Input x | Output f(x) | Auxiliary Qubit for computing the approximate biases and for making the case distinction in the registers | Auxiliary Walsh Approx Register with Qubit for Twos Complement |
        qubit_count = m + n + 1 + (approx_reg_qubit_count + 1)

        aux_qubit_index = m + n
        approx_reg_range = range(
            aux_qubit_index + 1, aux_qubit_index + 1 + (approx_reg_qubit_count + 1)
        )

        self.vbf = vbf
        self.m = m
        self.n = n
        self.tau = tau
        self.l = l
        self.d = d
        self.sample_inputs = sample_inputs
        self.sample_outputs = sample_outputs
        self.qubit_count = qubit_count
        self.approx_reg_qubit_count = approx_reg_qubit_count
        self.aux_qubit_index = aux_qubit_index
        self.approx_reg_range = approx_reg_range
        self.sampling_seed = sampling_seed
        self.precompute_success_prob = precompute_success_prob


@dataclass
class SITHAlgorithmResult(AlgorithmResult):
    """Result of a SITH execution containing extracted mask pairs.

    Attributes:
        mask_pairs: List of extracted (alpha, beta) mask pairs.
        iterations: Number of QAA iterations that were executed.
    """

    mask_pairs: list[tuple[int, int]] = field(default_factory=list)
    iterations: int = 0


class SITHAlgorithm(QuantumAlgorithm):
    """Stochastic Identification via Thresholding (SITH).

    Based on CEA and AA, this algorithm finds mask pairs of a VBF with minimal bias.
    """

    _ctx: SITHContext

    def __init__(self, executor: CircuitExecutor, ctx: SITHContext):
        """Initializes the SITH with an executor and a context.

        Args:
            executor: The CircuitExecutor instance used to run the circuit.
                NOTE: Setting shots != 1 leads to a false simulation, since the
                QAA-based algorithm can only sample once per iteration on a real
                quantum computer.
            ctx: The SITHContext to use, including the VectorialBooleanFunction
                to analyze.
        """
        super().__init__(executor)
        self._ctx = ctx

    def build_circuit(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        """Builds the CEA and Grover circuits for executing SITH via QAA.

        The CEA circuit is built in its full version, i.e., on the first m+n
        of self._ctx.qubit_count qubits.

        Returns:
            A tuple (cea_circuit_large, grover_circuit) containing the CEA
            circuit padded to the full qubit count and the corresponding Grover
            circuit.
        """
        ctx = self._ctx
        vbf, m, n, qubit_count = (ctx.vbf, ctx.m, ctx.n, ctx.qubit_count)
        cea_circuit = CorrelationExtraction(self._executor, vbf).build_circuit(
            measure=False
        )

        cea_circuit_large = QuantumCircuit(qubit_count)
        cea_circuit_large.compose(cea_circuit, range(m + n), inplace=True)

        grover_circuit = self._build_grover_circuit(cea_circuit)

        return (cea_circuit_large, grover_circuit)

    def run(self) -> SITHAlgorithmResult:
        """Executes the SITH algorithm and extracts mask pairs from measurements.

        Returns:
            A SITHAlgorithmResult with the extracted (alpha, beta) pairs, the
            underlying execution result and metadata, and the number of QAA
            iterations performed.
        """
        ctx = self._ctx

        cea_circuit_large, grover_circuit = self.build_circuit()

        success_prob = None
        if ctx.precompute_success_prob:
            success_prob = 0.0
            for a, b in product(range(1 << ctx.m), range(1 << ctx.n)):
                if self._verify_mask_pair_by_approximation((a, b, 0)):
                    success_prob += correlation(ctx.vbf, a, b) ** 2
            success_prob /= 1 << ctx.n

        aux_bitmask = (1 << (ctx.qubit_count - ctx.n - ctx.m)) - 1
        output_bitmask = (1 << (ctx.qubit_count - ctx.m)) - 1 - aux_bitmask
        input_bitmask = (1 << (ctx.qubit_count)) - 1 - output_bitmask
        qaa_config = QAAConfig(
            cea_circuit_large,
            grover_circuit,
            self._ctx.qubit_count,
            lambda x: self._verify_mask_pair_by_approximation(
                (
                    (x & input_bitmask) >> (ctx.qubit_count - ctx.m),
                    (x & output_bitmask) >> (ctx.qubit_count - ctx.n - ctx.m),
                    x & aux_bitmask,
                )
            ),
            success_prob,
            None,
            self._ctx.sampling_seed,
        )

        qaa = QuantumAmplitudeAmplification(self._executor, qaa_config)
        qaa_result = qaa.run()

        mask_pairs = self._verified_results_to_mask_pairs(qaa_result.verified_results)

        return SITHAlgorithmResult(
            execution_result=qaa_result.execution_result,
            metadata=qaa_result.metadata,
            mask_pairs=mask_pairs,
            iterations=qaa_result.iterations,
        )

    def _verified_results_to_mask_pairs(
        self, verified_results: list[int]
    ) -> list[tuple[int, int]]:
        """Converts verified QAA results to mask pairs.

        Args:
            verified_results: List of verified integer measurement outcomes
              produced by QAA.

        Returns:
            A list of (alpha, beta) tuples decoded from the verified results.
        """
        ctx = self._ctx
        m = ctx.m
        n = ctx.n
        mask_pairs: list[tuple[int, int]] = []

        for value in verified_results:
            value >>= ctx.qubit_count - n - m

            alpha = (value >> n) & ((1 << m) - 1)
            beta = value & ((1 << n) - 1)

            pair = (alpha, beta)
            mask_pairs.append(pair)

        return mask_pairs

    def _verify_mask_pair_by_approximation(self, mask_pair_and_aux):
        """Checks if a mask pair solves the search problem via approximation.

        Computes the approximated absolute bias of a mask pair and checks if it
        solves the search problem, also validating the state of the auxiliary
        qubits. In the article, corresponds to `Algorithm 2`.

        Args:
            mask_pair_and_aux: Tuple (alpha, beta, aux) of a mask pair and the
              auxiliary register value.

        Returns:
            True if the mask pair solves the search problem, False otherwise.
        """
        ctx = self._ctx
        sample_inputs = ctx.sample_inputs
        sample_outputs = ctx.sample_outputs
        tau = ctx.tau

        alpha = mask_pair_and_aux[0]
        beta = mask_pair_and_aux[1]
        aux = mask_pair_and_aux[2]

        approx_bias = approximate_bias(sample_inputs, sample_outputs, alpha, beta)
        return (alpha, beta) != (0, 0) and aux == 0 and abs(approx_bias) >= tau

    def _build_flip_zero_circuit(n):
        """Builds a circuit that flips an auxiliary qubit on the zero state.

        The auxiliary qubit is flipped if the first qubits correspond to the
        zero state. In the article, corresponds to `U_{chi_{(0, 0)}}`.

        Args:
            n: Number of qubits.

        Returns:
            An `n+1`-qubit circuit achieving the action described.
        """
        flip_zero_circuit = QuantumCircuit(n + 1)
        flip_zero_circuit.x(range(n))
        flip_zero_circuit.mcx(
            flip_zero_circuit.qubits[0:n],
            flip_zero_circuit.qubits[n],
            ctrl_state="1" * n,
        )
        flip_zero_circuit.x(range(n))
        return flip_zero_circuit

    @staticmethod
    def _build_increment_circuit(n, ctrl=False):
        """Builds a quantum circuit that increments the value represented.

        In the article, corresponds to `U_{Inc}`.

        Args:
            n: Number of qubits.
            ctrl: Whether to append a qubit and treat the zero'th qubit as a
              control.

        Returns:
            An `n`- or `n+1`-qubit circuit for incrementing a quantum register.
        """
        increment_circuit = QuantumCircuit(n + int(ctrl))
        for i in range(int(ctrl), n - 1 + int(ctrl)):
            if ctrl:
                increment_circuit.mcx(
                    list(
                        chain(
                            [increment_circuit.qubits[0]],
                            increment_circuit.qubits[(i + 1) : (n + 1)],
                        )
                    ),
                    increment_circuit.qubits[i],
                    ctrl_state="1" * (n - i + 1),
                )
            else:
                increment_circuit.mcx(
                    increment_circuit.qubits[(i + 1) : n],
                    increment_circuit.qubits[i],
                    ctrl_state="1" * (n - i - 1),
                )
        if ctrl:
            increment_circuit.cx(0, n)
        else:
            increment_circuit.x(increment_circuit.qubits[n - 1])
        return increment_circuit

    @staticmethod
    def _build_inner_product_circuit(n, x):
        """Builds a circuit that computes the binary inner product with x.

        Computes the binary inner product of a register with a given, fixed
        value representing a bitvector. In the article, corresponds to
        `U_{IP}(x, y)`.

        Args:
            n: Number of qubits.
            x: Value to compute the inner product with, action thus corresponds
              to <., x> computing.

        Returns:
            An `n+1`-qubit circuit achieving the action described. The last
            qubit stores the result.
        """
        #         Note:
        #             The circuit is very slightly different from the one in the paper as
        #             we assume one argument for the inner product and not two separate
        #             ones, though this has no semantic implications.
        assert isinstance(x, int) and 0 <= x < 1 << n
        inner_product_circuit = QuantumCircuit(n + 1)
        for i in range(n):
            if x == 0:
                break
            if x & 1 != 0:
                inner_product_circuit.cx(n - 1 - i, n)
            x >>= 1
        return inner_product_circuit

    @staticmethod
    def _build_inner_product_increment_circuit(n, d, x):
        """Builds a conditional increment circuit based on an inner product.

        If the binary inner product of a register with a given fixed value is
        zero, increments the value of an auxiliary register. In the article,
        corresponds to `U_{IPInc}(x, y)`.

        Args:
            n: Number of qubits in the register to compute the inner product
              with.
            x: Value to compute the inner product with, action thus corresponds
              to <., x> computing.
            d: Number of qubits in the auxiliary register storing a number.

        Returns:
            An `n+1+d`-qubit circuit achieving the action described. The
            `n+1`st qubit stores the result of the inner product, which also
            gets uncomputed.
        """
        assert isinstance(x, int) and 0 <= x < 1 << n
        inner_product_circuit = SITHAlgorithm._build_inner_product_circuit(n, x)
        increment_circuit = SITHAlgorithm._build_increment_circuit(d, ctrl=True)
        inner_product_increment_circuit = QuantumCircuit(n + 1 + d)
        inner_product_increment_circuit.compose(
            inner_product_circuit, qubits=range(n + 1), inplace=True
        )
        inner_product_increment_circuit.compose(
            increment_circuit, qubits=range(n, n + 1 + d), inplace=True
        )
        inner_product_increment_circuit.compose(
            inner_product_circuit, qubits=range(n + 1), inplace=True
        )
        return inner_product_increment_circuit

    @staticmethod
    def _build_integer_comparison_circuit(n, x, geq=False, ctrl=False):
        """Builds a circuit for comparing a positive integer with a fixed value.

        Precisely, the circuit computes whether the value in the register is
        smaller than the fixed value.

        Args:
            n: Number of qubits. Has to take into consideration that an
              additional bit is needed to make the comparison, since the two's
              complement is employed.
            x: Fixed number to compare with. It is assumed that
              0 <= x < 1<<(n-1).
            geq: Whether to compare for smaller by default or for greater or
              equal.
            ctrl: Whether to append a qubit and treat the zero'th qubit as a
              control.

        Returns:
            An `n+1`-qubit circuit for comparing the number in the first `n`
            qubits with `x` and that flips the `n+1`th bit based on that and
            whether `geq` is set or not. Adds an additional `0`th qubit for
            control, if `ctrl` is set to `True`.
        """
        circuit = QuantumCircuit(n + 1 + int(ctrl))
        res_qubit_index = n + int(ctrl)
        assert 0 <= x < 1 << (n - 1)

        y = int_to_twos_complement_repr(-x, n)
        z = int_to_twos_complement_repr(x, n)

        for i in range(n):
            if y & (1 << i) != 0:
                circuit.compose(
                    SITHAlgorithm._build_increment_circuit(n - i),
                    qubits=range(int(ctrl), n - i + int(ctrl)),
                    inplace=True,
                )

        if ctrl:
            circuit.mcx(circuit.qubits[0:2], res_qubit_index, ctrl_state="11")
            if geq:
                circuit.cx(0, res_qubit_index)
        else:
            circuit.mcx([circuit.qubits[0]], res_qubit_index)
            if geq:
                circuit.x(res_qubit_index)

        for i in range(n):
            if z & (1 << i) != 0:
                circuit.compose(
                    SITHAlgorithm._build_increment_circuit(n - i),
                    qubits=range(int(ctrl), n - i + int(ctrl)),
                    inplace=True,
                )

        return circuit

    @staticmethod
    def _build_integer_comparison_circuit_leq(n, x):
        """Builds a circuit for comparing a positive integer for less-or-equal.

        Compares a positive integer with a fixed value, evaluating for smaller
        or equal. In the article, corresponds to `U_{leq}(x)`.

        Args:
            n: Number of qubits.
            x: Fixed number to compare with.

        Returns:
            An `n+1`-qubit circuit for comparing the number in the first `n`
            qubits with `x` and that flips the `n+1`th bit based on that.
        """
        return SITHAlgorithm._build_integer_comparison_circuit(
            n, (x + 1) % (1 << (n - 1)), geq=(x == (1 << (n - 1)) - 1)
        )

    @staticmethod
    def _build_integer_comparison_circuit_geq(n, x):
        """Builds a circuit for comparing a positive integer for greater-or-equal.

        Compares a positive integer with a fixed value, evaluating for greater
        or equal. In the article, corresponds to `U_{geq}(x)`.

        Args:
            n: Number of qubits.
            x: Fixed number to compare with.

        Returns:
            An `n+1`-qubit circuit for comparing the number in the first `n`
            qubits with `x` and that flips the `n+1`th bit based on that.
        """
        return SITHAlgorithm._build_integer_comparison_circuit(n, x, geq=True)

    def _build_walsh_approximation_summation_circuit(self):
        """Builds the summation circuit for the Walsh approximation circuit.

        Returns:
            The `QuantumCircuit` representing the associated Walsh
            approximation summation circuit.
        """
        ctx = self._ctx
        m = ctx.m
        n = ctx.n
        sample_inputs = ctx.sample_inputs
        sample_outputs = ctx.sample_outputs
        qubit_count = ctx.qubit_count
        aux_qubit_index = ctx.aux_qubit_index
        approx_reg_qubit_count = ctx.approx_reg_qubit_count

        approx_summation_circuit = QuantumCircuit(qubit_count)
        approx_summation_circuit.x(aux_qubit_index)
        for x, y in zip(sample_inputs, sample_outputs):
            # For the input, compute whether the inner product is even and store that in the auxiliary qubit, whether the inner product is even and store that in the auxiliary qubit.
            # Compute the sign of the inner product sum, note that `mcx` cannot be used here since we wish to know if the inner product is even.
            # The `ModularAdderGate` only works on two registers of same bit length. To save bits, we do a custom implementation to add a qubit to an n-qubit register.
            approx_summation_circuit.compose(
                SITHAlgorithm._build_inner_product_increment_circuit(
                    m + n, approx_reg_qubit_count + 1, (x << n) | y
                ),
                qubits=range(qubit_count),
                inplace=True,
            )
        approx_summation_circuit.x(aux_qubit_index)
        return approx_summation_circuit

    def _build_walsh_approximation_marking_circuit(self):
        """Builds the part of the Walsh approximation circuit that marks masks.

        Marks all masks that are considered sufficient under the Walsh
        approximation.

        Returns:
            The `QuantumCircuit` representing the associated Walsh
            approximation marking circuit.
        """
        ctx = self._ctx
        m = ctx.m
        n = ctx.n
        tau = ctx.tau
        sample_inputs = ctx.sample_inputs
        qubit_count = ctx.qubit_count
        approx_reg_qubit_count = ctx.approx_reg_qubit_count
        aux_qubit_index = ctx.aux_qubit_index
        approx_reg_range = ctx.approx_reg_range
        s = len(sample_inputs)

        approx_marking_circuit = QuantumCircuit(qubit_count)
        approx_marking_circuit.compose(
            self._build_walsh_approximation_summation_circuit(),
            range(qubit_count),
            inplace=True,
        )

        # Perform a case distinction to compare the absolute value of the approximation register with tau appropriately
        c1 = ceil(s * (tau + 1 / 2))
        approx_marking_circuit.compose(
            SITHAlgorithm._build_integer_comparison_circuit_geq(
                approx_reg_qubit_count + 1, c1
            ),
            qubits=chain(approx_reg_range, [aux_qubit_index]),
            inplace=True,
        )

        c2 = floor(s * (1 / 2 - tau))
        approx_marking_circuit.compose(
            SITHAlgorithm._build_integer_comparison_circuit_leq(
                approx_reg_qubit_count + 1, c2
            ),
            qubits=chain(approx_reg_range, [aux_qubit_index]),
            inplace=True,
        )

        # Exclude the zero
        approx_marking_circuit.compose(
            SITHAlgorithm._build_flip_zero_circuit(m + n),
            qubits=range(m + n + 1),
            inplace=True,
        )

        return approx_marking_circuit

    def _build_walsh_approximation_circuit(self):
        """Builds the Walsh approximation circuit for a SITH problem.

        Negates the amplitude of elements solving the search problem under
        Walsh approximation.

        Returns:
            The `QuantumCircuit` representing the associated Walsh
            approximation circuit.
        """
        ctx = self._ctx
        qubit_count = ctx.qubit_count
        aux_qubit_index = ctx.aux_qubit_index

        approx_circuit = QuantumCircuit(qubit_count)
        approx_circuit.compose(
            self._build_walsh_approximation_marking_circuit(),
            range(qubit_count),
            inplace=True,
        )
        approx_circuit.z(aux_qubit_index)
        approx_circuit.compose(
            self._build_walsh_approximation_marking_circuit().inverse(),
            range(qubit_count),
            inplace=True,
        )

        return approx_circuit

    @staticmethod
    def _build_global_negate_circuit():
        """Builds a 1-qubit circuit for negating the phases.

        Sufficient for negating the phases of larger circuits. Requires the
        associated qubit to be zeroed.

        Returns:
            The `QuantumCircuit` representing the associated circuit.
        """
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.z(0)
        circuit.x(0)
        return circuit

    def _build_grover_circuit(self, cea_circuit):
        """Builds the Grover circuit for a SITH problem.

        Args:
            cea_circuit: The CEA `QuantumCircuit` to use.

        Returns:
            The `QuantumCircuit` representing the associated Grover circuit.
        """
        ctx = self._ctx
        m = ctx.m
        n = ctx.n
        qubit_count = ctx.qubit_count
        aux_qubit_index = ctx.aux_qubit_index

        # We then build the Grover operator and append it as many times as needed
        grover_circuit = QuantumCircuit(qubit_count)

        # Negate desired mask pairs
        grover_circuit.compose(
            self._build_walsh_approximation_circuit(),
            qubits=range(qubit_count),
            inplace=True,
        )
        # `CEA^\dagger`
        grover_circuit.compose(cea_circuit.inverse(), qubits=range(m + n), inplace=True)
        # Negate 0
        grover_circuit.compose(
            build_phase_flip_x_circuit(m + n, 0),
            range(aux_qubit_index + 1),
            inplace=True,
        )
        # `CEA`
        grover_circuit.compose(cea_circuit, qubits=range(m + n), inplace=True)
        # Global -1
        grover_circuit.compose(
            self._build_global_negate_circuit(), qubits=[aux_qubit_index], inplace=True
        )

        return grover_circuit
