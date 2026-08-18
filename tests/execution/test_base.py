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
import pytest
from qiskit import QuantumCircuit

from qace.execution import ExecutionConfig

# === Tests: ExecutionConfig Initialization ===


class TestExecutionConfigInit:

    def test_default_values(self):
        """Default config has empty dicts."""
        config = ExecutionConfig()
        assert config.run_options == {}
        assert config.transpiler_options == {}
        assert config.backend_options == {}

    def test_custom_run_options(self):
        """Custom run_options are stored correctly."""
        config = ExecutionConfig(run_options={"shots": 2048})
        assert config.run_options == {"shots": 2048}

    def test_custom_transpiler_options(self):
        """Valid transpiler_options are stored correctly."""
        config = ExecutionConfig(transpiler_options={"optimization_level": 3})
        assert config.transpiler_options == {"optimization_level": 3}

    def test_custom_backend_options(self):
        """Custom backend_options are stored correctly."""
        config = ExecutionConfig(backend_options={"method": "statevector"})
        assert config.backend_options == {"method": "statevector"}

    def test_all_options_combined(self):
        """All options can be set simultaneously."""
        config = ExecutionConfig(
            run_options={"shots": 1024},
            transpiler_options={"optimization_level": 2},
            backend_options={"method": "density_matrix"},
        )
        assert config.run_options["shots"] == 1024
        assert config.transpiler_options["optimization_level"] == 2
        assert config.backend_options["method"] == "density_matrix"


# === Tests: Instance Isolation ===


class TestExecutionConfigIsolation:

    def test_run_options_not_shared_between_instances(self):
        """Each instance gets its own run_options dict."""
        config1 = ExecutionConfig()
        config2 = ExecutionConfig()

        config1.run_options["shots"] = 999
        assert "shots" not in config2.run_options

    def test_transpiler_options_not_shared_between_instances(self):
        """Each instance gets its own transpiler_options dict."""
        config1 = ExecutionConfig()
        config2 = ExecutionConfig()

        config1.transpiler_options["optimization_level"] = 3
        assert "optimization_level" not in config2.transpiler_options

    def test_backend_options_not_shared_between_instances(self):
        """Each instance gets its own backend_options dict."""
        config1 = ExecutionConfig()
        config2 = ExecutionConfig()

        config1.backend_options["method"] = "statevector"
        assert "method" not in config2.backend_options


# === Tests: Validation ===


class TestExecutionConfigValidation:

    def test_circuit_key_forbidden(self):
        """'circuit' in transpiler_options raises ValueError."""
        qc = QuantumCircuit(1)

        with pytest.raises(ValueError, match="not allowed in transpiler_options"):
            ExecutionConfig(transpiler_options={"circuit": qc})

    def test_circuits_key_forbidden(self):
        """'circuits' in transpiler_options raises ValueError."""
        qc = QuantumCircuit(1)

        with pytest.raises(ValueError, match="not allowed in transpiler_options"):
            ExecutionConfig(transpiler_options={"circuits": [qc]})

    def test_error_message_contains_offending_key(self):
        """Error message includes the forbidden key that was found."""
        qc = QuantumCircuit(1)

        with pytest.raises(ValueError, match="circuit"):
            ExecutionConfig(transpiler_options={"circuit": qc})

    def test_both_forbidden_keys_reported(self):
        """Both 'circuit' and 'circuits' are reported if both present."""
        qc = QuantumCircuit(1)

        with pytest.raises(ValueError, match="not allowed in transpiler_options"):
            ExecutionConfig(transpiler_options={"circuit": qc, "circuits": [qc]})

    def test_valid_keys_not_rejected(self):
        """Keys other than 'circuit'/'circuits' are allowed."""
        config = ExecutionConfig(
            transpiler_options={
                "optimization_level": 2,
                "seed_transpiler": 42,
                "coupling_map": [[0, 1]],
                "backend": None,
            }
        )
        assert config.transpiler_options["optimization_level"] == 2
        assert config.transpiler_options["seed_transpiler"] == 42


# === Tests: skip_transpilation field ===


class TestExecutionConfigSkipTranspilation:

    def test_default_is_false(self):
        assert ExecutionConfig().skip_transpilation is False

    def test_can_be_set_true(self):
        assert ExecutionConfig(skip_transpilation=True).skip_transpilation is True
