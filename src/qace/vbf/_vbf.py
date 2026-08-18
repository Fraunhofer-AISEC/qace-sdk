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
"""Vectorial Boolean function abstraction and full precomputed implementation.

This module defines the abstract base class for vectorial Boolean functions
and a concrete implementation that precomputes and stores all function values.
It provides methods for evaluation, quantum circuit construction, and sparse
unitary generation.
"""

# qace-sdk
# vbf/_vbf.py
# Fraunhofer AISEC

from abc import ABC, abstractmethod
from collections.abc import Callable
from qace.vbf._circuit_conventions import mcx_msb0
from bitarray import bitarray
from functools import reduce
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.boolean.boolean_expression import TruthTable
from qiskit.synthesis.boolean.boolean_expression_synth import EsopGenerator
from scipy.sparse import coo_array, csc_array

MAX_BIT_COUNT = 18


class VectorialBooleanFunction(ABC):
    """
    Class for modelling vectorial Boolean functions.
    """

    @abstractmethod
    def __init__(self, f: Callable[[int], int], m: int, n: int) -> None:
        """Initializes a vectorial Boolean function.

        Args:
            f: A vectorial Boolean function from `m` to `n` bits, which itself is a function from `int` to `int`.
            m: Domain bit count.
            n: Image bit count.

        Returns:
            The initialized function.
        """
        self.m = m
        self.n = n
        self.M = 1 << self.m
        self.N = 1 << self.n
        self._f = f

    @abstractmethod
    def eval(self, x: int) -> int:
        """Evaluates the vectorial Boolean function at a point `x`.

        Args:
            x: Input argument `x`.

        Returns:
            Value `f(x)`.
        """
        pass

    def to_circuit(self, method: str = "qiskit_esop_generator") -> QuantumCircuit:
        """Constructs a quantum circuit representing the Boolean function using the specified method. Override for custom circuit implementations.

        Args:
            method: "minterms" to use the minterm-based approach (truth table) or
                    "qiskit_esop_generator" to use the qiskit esop_generator class, that uses Shannon's expansion
                    based term minimization.

        Returns:
            The Qiskit quantum circuit encoding the Boolean function.
        """
        if method == "minterms":
            return self._to_circuit_minterms()
        elif method == "qiskit_esop_generator":
            return self._to_ciruit_qiskit_esop_generator()
        else:
            ValueError(
                f'method must be "minterms" or "qiskit_esop_generator" found: {method}'
            )

    def to_sparse_unitary(self) -> csc_array:
        """Creates a `scikit.sparse.coo_array` corresponding to an oracle unitary for the function.

        Returns:
            Sparsely encoded unitary oracle corresponding to the function.
        """

        # Custom `to_gate` function, as `tweedledum` does not build with `pip` for some reason, as of 05.07.2024, 00:14 with version v1.1.1.,
        # so Qiskits `classical_function` API cannot be used.
        # See also https://docs.quantum.ibm.com/api/qiskit/classicalfunction and https://github.com/boschmitt/tweedledum/issues/186.
        def col_generator(f: VectorialBooleanFunction):
            for x in range(1 << f.m):
                for y in range(1 << f.n):
                    # Determines index for setting one according to the oracle description |x>|y> |-> |x>|y + f(x)>.
                    yield (x << f.n) | (y ^ f.eval(x))

        size = 1 << (self.m + self.n)
        unitary = coo_array(
            ([1] * size, (list(col_generator(self)), range(size))),
            shape=(size, size),
            dtype=complex,
        ).tocsc()
        return unitary

    def _to_circuit_minterms(self) -> QuantumCircuit:
        """Constructs a quantum circuit based on the minterms (truth table) of the Boolean function.
        The circuit implements an oracle unitary using AND gates to detect minterms and controlled-X to set outputs.
        Assumes an auxiliary qubit (m+n+1-th) initialized to |0>.

        Returns:
            A Qiskit circuit that implements the minterm-based oracle for the Boolean function.
        """
        m = self.m
        n = self.n

        circuit = QuantumCircuit(m + n)
        for x in range(1 << m):
            fx = self.eval(x)
            for j in range(n):
                if fx & (1 << j) != 0:
                    mcx_msb0(
                        circuit,
                        control_qubits=circuit.qubits[0:m],
                        target_qubit=circuit.qubits[(m + n - 1) - j],
                        ctrl_state=x,
                    )

        circuit = circuit.decompose()
        return circuit

    def _to_ciruit_qiskit_esop_generator(self) -> QuantumCircuit:
        """Constructs a quantum circuit using Qiskit's ESOP generator.

        Uses Shannon's expansion-based term minimization to synthesize
        the Boolean function into a quantum circuit with multi-controlled
        X gates.

        Returns:
            A Qiskit circuit that implements the ESOP-based oracle for the
            Boolean function.
        """
        circuit = QuantumCircuit(self.m + self.n)
        esop_list = self._gen_esop_list()
        for j, esops in enumerate(esop_list):
            for esop in esops:
                if esop == "-" * self.n:
                    circuit.x((self.m + self.n - 1) - j)
                    continue
                input_bits = []
                for bit_index, c in enumerate(esop):
                    if c == "-":
                        continue
                    input_bits.append(bit_index)
                values = esop.replace("-", "")
                # The bit ordering convention in this project is big-endian, e. g., in "10000" the msb is 1 or at index 0
                # Qubit order x_0 x_1 ... x_m y_0 y_1 ... y_n ; for x denotes x_0 the msb and for y denotes y_0 msb
                # The values in the esop use big-endian. Since ctrl_state uses little-endian, we need to reverse values.
                # The input_bits use big-endian, thus, we can use them straight up.
                # Since the espo_list starts with the lsb, we need to count the target qubit beginning with the highest index value.
                mcx_msb0(
                    circuit,
                    control_qubits=input_bits,
                    target_qubit=(self.m + self.n - 1) - j,
                    ctrl_state=values,
                )
        return circuit

    def _gen_esop_list(self) -> list[list[str]]:
        """Generates ESOP representations for all output bits of the Boolean function.

        For each output index j, this method constructs a single-output truth table
        from the multi-output Boolean function, runs the ESOP generator on it, and
        collects the resulting ESOP expressions.

        Returns:
            A list of ESOP representations, where each entry contains the ESOP for
            one output bit of the Boolean function.
        """
        esop_list = []
        for j in range(self.n):

            def f(x):
                xint = reduce(lambda acc, b: (acc << 1) | b, map(int, x), 0)
                return bool((self.__call__(xint) >> j) & 1)

            tt = TruthTable(f, self.m)
            esop_gen = EsopGenerator(tt)
            esops = esop_gen.esop
            esop_list.append(esops)

        return esop_list


class VectorialBooleanFunctionFull(VectorialBooleanFunction):
    """A fully precomputed vectorial Boolean function.

    Stores all output values of the function in a bitarray for efficient
    evaluation without recomputation.

    Attributes:
        f: A bitarray storing the precomputed output values of the function.
        m: The number of input bits.
        n: The number of output bits.
        M: The size of the input domain (2^m).
        N: The size of the output domain (2^n).
    """

    def __init__(self, f: Callable[[int], int], m: int, n: int) -> None:
        """Initializes a vectorial Boolean function. Precomputes and stores all values of `f` using the given function object.

        Args:
            f: A vectorial Boolean function from `m` to `n` bits, which itself is a function from `int` to `int`.
            m: Domain bit count.
            n: Image bit count.

        Returns:
            The initialized function.
        """
        assert m <= MAX_BIT_COUNT and n <= MAX_BIT_COUNT
        super().__init__(f, m, n)
        self.f = bitarray((1 << m) * n)
        for x in range(1 << m):
            self.f[x * n : (x + 1) * n] = bitarray(format(f(x), f"#0{n+2}b")[2:])
        self.__call__ = lambda x: int(self.f[(x * n) : ((x + 1) * n)].to01(), 2)

    def eval(self, x: int) -> int:
        """Evaluates the vectorial Boolean function at a point `x`.

        Args:
            x: Input argument `x`.

        Returns:
            Value `f(x)`.
        """
        return self.__call__(x)
