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
"""Tests for VBF circuit construction and execution equivalence."""

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qace.execution import AerExecutor, ExecutionConfig
from qace.vbf import VectorialBooleanFunctionFull, RandomVBF, measure_msb0
import pytest


@pytest.fixture
def identity_3bit_vbf():
    """Provides the identity VBF on 3 bits, i.e., ``f(x) = x``."""
    m = 3
    n = 3
    return VectorialBooleanFunctionFull(lambda x: x, m, n)


@pytest.fixture
def executor():
    """Provides a shared default ``AerExecutor`` to avoid repeated init cost."""
    return AerExecutor.default()


class TestIdentity2BitVBF:
    """Tests for the identity 3-bit VBF circuit correctness."""

    def test_circuit_applies_identity_qiskit_esop_generator(
        self, identity_3bit_vbf, executor
    ):
        """Verifies the ESOP circuit implements ``|x>|y> -> |x>|y ^ x>``."""
        m = 3
        n = 3
        vbf = identity_3bit_vbf

        # Build circuit
        circuit = vbf.to_circuit(method="qiskit_esop_generator")

        # Expected: circuit has m + n qubits, no classical bits
        assert circuit.num_qubits == m + n
        assert circuit.num_clbits == 0

        # Test for all possible inputs x ∈ {0,1,2,3} with y=0
        for x in range(1 << m):
            # Prepare |x>|0> on full register (4 qubits)
            qc = QuantumCircuit(m + n, m + n)
            qc.initialize(x, qubits=list(reversed(range(m))))
            # Remaining n qubits are |0⟩ by default

            # Apply VBF circuit
            qc.compose(circuit, inplace=True)

            # Measure *entire* register (m + n qubits)
            measure_msb0(qc, range(m + n), range(m + n))

            result = executor.execute(qc)
            measured = int(list(result.counts.keys())[0], 2)

            # Expected: |x>|0 ⊕ x> = |x>|x> → binary index = (x << n) | x = x * 4 + x = 5*x
            expected = (x << n) | x  # = 5*x for n=2
            print(format(measured, f"#0{(m+n)+2}b")[2:])
            print(format(expected, f"#0{(m + n) + 2}b")[2:])
            assert measured == expected, (
                f"For input x={x}, expected full register state |x>|f(x)> = |{x:02b}>|{x:02b}> "
                f"(index {expected}), got index {measured} ({bin(measured)[2:].zfill(4)})"
            )

    def test_circuit_applies_identity_minterms(self, identity_3bit_vbf, executor):
        """Verifies the minterms circuit implements ``|x>|y> -> |x>|y ^ x>``."""
        m = 3
        n = 3
        vbf = identity_3bit_vbf

        # Build circuit
        circuit = vbf.to_circuit(method="minterms")

        # Expected: circuit has m + n qubits, no classical bits
        assert circuit.num_qubits == m + n
        assert circuit.num_clbits == 0

        # Test for all possible inputs x ∈ {0,1,2,3} with y=0
        for x in range(1 << m):
            # Prepare |x>|0> on full register (4 qubits)
            qc = QuantumCircuit(m + n, m + n)
            qc.initialize(x, qubits=list(reversed(range(m))))
            # Remaining n qubits are |0⟩ by default

            # Apply VBF circuit
            qc.compose(circuit, inplace=True)

            # Measure *entire* register (m + n qubits)
            measure_msb0(qc, range(m + n), range(m + n))

            result = executor.execute(qc)
            measured = int(list(result.counts.keys())[0], 2)

            # Expected: |x>|0 ⊕ x> = |x>|x> → binary index = (x << n) | x = x * 4 + x = 5*x
            expected = (x << n) | x  # = 5*x for n=2
            assert measured == expected, (
                f"For input x={x}, expected full register state |x>|f(x)> = |{x:02b}>|{x:02b}> "
                f"(index {expected}), got index {measured} ({bin(measured)[2:].zfill(4)})"
            )


class TestVBF:
    """Test suite for general VBF-to-circuit compilation and execution."""

    def test_vbf_to_unitary_gate(self):
        """Verifies both compilation methods realize the VBF oracle on all
        basis states ``|x>|y>``.
        """
        lt = [2, 6, 5, 4, 2, 0, 7, 0]

        def f(x):
            """Returns the lookup-table value ``lt[x]``."""
            return lt[x]

        vbf = VectorialBooleanFunctionFull(f, 3, 3)
        m, n = (vbf.m, vbf.n)
        vbf_circuit_minterms = vbf.to_circuit(method="minterms")
        vbf_circuit_qiskit_esop_generator = vbf.to_circuit(
            method="qiskit_esop_generator"
        )

        # due to circuit.initialize we need an executor with transpilation here
        executor = AerExecutor(ExecutionConfig())

        def _test_circuit(vbf_circuit, x, y):
            """Asserts the circuit maps ``|x>|y>`` to ``|x>|y ^ f(x)>``."""
            circuit = QuantumCircuit(m + n, n)
            circuit.initialize((x << m) | y, qubits=list(reversed(range(m + n))))
            circuit.append(vbf_circuit, qargs=range(m + n))
            measure_msb0(circuit, range(m, m + n), range(n))
            execution_results = executor.execute(circuit)
            result, *_ = execution_results.counts.keys()
            assert int(result, 2) == y ^ vbf.eval(x)

        # Test the action on the canonical basis vectors.
        for x in range(1 << m):
            for y in range(1 << n):
                _test_circuit(vbf_circuit_minterms, x, y)
                _test_circuit(vbf_circuit_qiskit_esop_generator, x, y)

    def test_3_to_3_random_vbfs_qiskit_esop_generator(self, executor):
        """Verifies the ESOP circuit for 10 random 3-to-3 VBFs on all inputs."""
        m = 3
        n = 3

        for i in range(10):
            vbf = RandomVBF(m, n)
            vbf_circuit = vbf.to_circuit(method="qiskit_esop_generator")
            for y in range(1 << n):
                for x in range(1 << m):
                    reg_x = QuantumRegister(m, "x")
                    reg_y = QuantumRegister(n, "y")
                    reg_mes = ClassicalRegister(n, "reg_mes")
                    circuit = QuantumCircuit(reg_x, reg_y, reg_mes)
                    circuit.initialize(x, qubits=list(reversed(range(m))))
                    circuit.initialize(y, qubits=list(reversed(range(m, m + n))))
                    circuit.compose(vbf_circuit, inplace=True)
                    measure_msb0(circuit, range(m, m + n), reg_mes)

                    execution_results = executor.execute(circuit)
                    result, *_ = execution_results.counts.keys()

                    circuit_y = int(result, 2)
                    assert circuit_y == vbf.eval(x) ^ y


@pytest.fixture
def random_circuits():
    """Produces pairs of minterms/ESOP circuits for random VBFs."""

    def _random_circuits(m, n):
        """Builds 5 seeded VBFs and returns (minterms, ESOP) circuit pairs."""
        vbf_base_seed = 0xD5AC01A
        circuits = []
        for i in range(5):
            vbf = RandomVBF(m, n, rng_seed=vbf_base_seed + i)
            circuit_minterms = vbf.to_circuit(method="minterms")
            circuit_esop = vbf.to_circuit(method="qiskit_esop_generator")
            circuits.append((circuit_minterms, circuit_esop))
        return circuits

    return _random_circuits


@pytest.fixture
def x_zero_circuit():
    """Builds a Hadamard-superposed input circuit around a VBF."""

    def _x_zero_circuit(m, n, vbf_circuit, x):
        """Prepares ``|x>`` with Hadamards on the input register, applies the
        VBF circuit, and measures the full register.
        """
        qc = QuantumCircuit(m + n, m + n)
        qc.initialize(x, qubits=list(reversed(range(m))))
        qc.h(range(m))
        qc.compose(vbf_circuit, inplace=True)
        measure_msb0(qc, range(m + n), range(m + n))
        return qc

    return _x_zero_circuit


@pytest.fixture
def x_zero_circuit_super():
    """Builds an input circuit around a VBF and measures the full register."""

    def _x_zero_circuit(m, n, vbf_circuit, x):
        """Prepares ``|x>``, applies the VBF circuit, and measures the full
        register.
        """
        qc = QuantumCircuit(m + n, m + n)
        qc.initialize(x, qubits=list(reversed(range(m))))
        qc.compose(vbf_circuit, inplace=True)
        measure_msb0(qc, range(m + n), range(m + n))
        return qc

    return _x_zero_circuit


@pytest.fixture
def shots_100_det_executor():
    """Deterministic ``AerExecutor`` fixed at 100 shots and seed 42."""
    config = ExecutionConfig(
        run_options={"shots": 100},
        backend_options={"seed_simulator": 42},
        skip_transpilation=True,
    )
    return AerExecutor(config=config)


class TestMethodEqualityExecution:
    """Verifies both compilation methods yield identical measurements."""

    def test_methods_agree(
        self, random_circuits, x_zero_circuit, shots_100_det_executor
    ):
        """Asserts both circuits yield identical counts on all inputs."""
        m = 4
        n = m
        for circuit_minterms, circuit_esop in random_circuits(m, n):
            for x in range(1 << m):
                res_minterms = shots_100_det_executor.execute(
                    x_zero_circuit(m, n, circuit_minterms, x)
                ).counts
                res_esop = shots_100_det_executor.execute(
                    x_zero_circuit(m, n, circuit_esop, x)
                ).counts
                assert res_minterms == res_esop

    def test_methods_agree_super(
        self, random_circuits, x_zero_circuit_super, shots_100_det_executor
    ):
        """Asserts agreement of both circuits under the superposed fixture."""
        m = 4
        n = m
        for circuit_minterms, circuit_esop in random_circuits(m, n):
            for x in range(1 << m):
                res_minterms = shots_100_det_executor.execute(
                    x_zero_circuit_super(m, n, circuit_minterms, x)
                ).counts
                res_esop = shots_100_det_executor.execute(
                    x_zero_circuit_super(m, n, circuit_esop, x)
                ).counts
                assert res_minterms == res_esop
