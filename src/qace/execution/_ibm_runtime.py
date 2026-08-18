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
"""IBM Quantum Runtime circuit executor implementation.

Provides an executor that leverages IBM Runtime's [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) primitive
for running quantum circuits on both local fake backends and real
cloud backends.
"""

import time, copy
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2

from qace.execution._base import CircuitExecutor, ExecutionResult, ExecutionConfig


@dataclass
class IBMRuntimeExecutionConfig(ExecutionConfig):
    """ExecutionConfig extended with [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) options.

    Available options for the inherited fields can be found in the
    respective Qiskit documentation:

    - ``run_options``: Supports ``"shots"`` (default 1024).
    - ``transpiler_options``: See
      ``qiskit.generate_preset_pass_manager()``
      [documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.generate_preset_pass_manager).
      Note that ``"backend"`` must not be included.
    - ``backend_options``: Must contain a ``"backend"`` key with a
      [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance.
    - ``sampler_options``: See ``qiskit_ibm_runtime.SamplerV2.options``
      [documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-sampler-options).

    Attributes:
        sampler_options: Options passed to the [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) instance.
    """

    sampler_options: dict[str, Any] = field(default_factory=dict)


class IBMRuntimeExecutor(CircuitExecutor):
    """A circuit executor that uses the IBM Runtime [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) primitive.

    This executor wraps the IBM Quantum Runtime sampling pipeline. It works
    identically with local fake backends (e.g. from ``qiskit_ibm_runtime.fake_provider``)
    and real IBM Quantum cloud backends. The setup is split between
    construction time and execution time:

    During ``__init__``:

    ```python
    # Step 1: Backend extraction
    backend = config.backend_options["backend"]  # a BackendV2 instance

    # Step 2: Pass manager construction
    pass_manager = generate_preset_pass_manager(
        backend=backend, **config.transpiler_options
    )

    # Step 3: Sampler construction
    sampler = SamplerV2(mode=backend)
    sampler.options.update(**config.sampler_options)
    ```

    During each ``execute()`` call:

    ```python
    # Step 4: Transpilation
    transpiled = pass_manager.run(circuit)

    # Step 5: Execution via SamplerV2
    shots = config.run_options.get("shots", 1024)
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    counts = result[0].data.<creg>.get_counts()
    ```

    ## Understanding `IBMRuntimeExecutionConfig` fields for `IBMRuntimeExecutor`

    **`backend_options`** — must contain a ``"backend"`` key with a [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance:

    The [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance determines both the target device topology
    (coupling map, basis gates) used during transpilation and the noise
    model used during simulation (for fake backends). No additional keys
    are required, but the ``"backend"`` key is mandatory.

    ```python
    backend_options = {
        "backend": FakeManilaV2(),   # REQUIRED: any BackendV2 instance
    }
    ```

    Full list of compatible backends: any class implementing the Qiskit
    [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) interface, including fake backends from
    ``qiskit_ibm_runtime.fake_provider`` and real backends obtained via
    ``QiskitRuntimeService`` (see [external link](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-service)).

    **`run_options`** — control per-execution parameters:

    Currently, the only supported key is ``"shots"``, which determines
    the number of measurement samples per execution. If omitted, the
    default is 1024 shots.

    ```python
    run_options = {
        "shots": 4096,   # number of measurement samples (default: 1024)
    }
    ```

    **`transpiler_options`** — passed to
    ``qiskit.generate_preset_pass_manager(...)`` as kwargs:

    These control how the quantum circuit is compiled/optimized before
    execution. The ``backend`` kwarg is automatically set to the
    configured [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance; do NOT include it manually. The pass
    manager is built once during ``__init__`` and reused for every
    ``execute()`` call.

    The following defaults are applied if not explicitly provided:

    ```python
    transpiler_options = {
        "optimization_level": 0,             # minimal optimization
        "layout_method": "trivial",          # identity qubit mapping
        "routing_method": "basic",           # simple swap insertion
        "translation_method": "translator",  # basis gate translation
    }
    ```

    Common examples for overriding:

    ```python
    transpiler_options = {
        "optimization_level": 2,   # 0-3, higher = more optimization
        "seed_transpiler": 42,     # for reproducible transpilation
        "coupling_map": None,      # override backend coupling map
    }
    ```

    Full list: see the [documentation](https://docs.quantum.ibm.com/api/qiskit/transpiler_preset)
    of ``qiskit.generate_preset_pass_manager()``.

    **`sampler_options`** — passed to ``SamplerV2.options.update(...)``
    after construction:

    These configure the [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) primitive instance (e.g. simulator
    seed for reproducibility on fake backends, execution mode settings).
    They are set once at construction time and persist across all
    ``execute()`` calls (unless temporarily overridden by
    ``simulator_seed``).

    Common examples:

    ```python
    sampler_options = {
        "simulator": {
            "seed_simulator": 42,   # for reproducible simulation results
        },
    }
    ```

    Full list: see ``qiskit_ibm_runtime.options.SamplerOptions``
    [documentation](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.options.SamplerOptions).

    ## Example usage

    ```python
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qace.execution import IBMRuntimeExecutor, IBMRuntimeExecutionConfig

    config = IBMRuntimeExecutionConfig(
        backend_options={"backend": FakeManilaV2()},
        run_options={"shots": 4096},
        transpiler_options={"optimization_level": 2, "seed_transpiler": 42},
        sampler_options={"simulator": {"seed_simulator": 42}},
    )
    executor = IBMRuntimeExecutor(config)
    result = executor.execute(circuit)
    print(result.counts)
    ```
    """

    def __init__(self, config: IBMRuntimeExecutionConfig | None = None):
        """Initializes the executor with the given configuration.

        Constructs the preset pass manager and the [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) instance at
        construction time so they can be reused across multiple
        ``execute()`` calls:

        ```python
        self._passmanager = generate_preset_pass_manager(
            backend=self._backend, **config.transpiler_options
        )
        self._sampler = SamplerV2(mode=self._backend)
        self._sampler.options.update(**config.sampler_options)
        ```

        The pass manager is configured with the following defaults if not
        explicitly specified in ``config.transpiler_options``:

        - ``optimization_level``: 0
        - ``layout_method``: "trivial"
        - ``routing_method``: "basic"
        - ``translation_method``: "translator"

        Args:
            config: An IBMRuntimeExecutionConfig instance. If None, a
                default configuration is used.

        Raises:
            RuntimeError: If 'backend' is missing from
                backend_options or is not a [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance.
        """
        super().__init__()
        self.config = copy.deepcopy(config) or IBMRuntimeExecutionConfig()
        backend = self.config.backend_options.get("backend")

        if backend is None:
            raise RuntimeError(
                "backend_options must contain a 'backend' key with "
                "a BackendV2 instance."
            )
        if not isinstance(backend, BackendV2):
            raise RuntimeError(
                f"backend_options['backend'] must be a BackendV2 instance, "
                f"got {type(backend).__name__}."
            )

        self._backend = backend

        # setup pass manager for circuit transpilation
        self.config.transpiler_options["backend"] = self._backend
        self.config.transpiler_options.setdefault("optimization_level", 0)
        self.config.transpiler_options.setdefault("layout_method", "trivial")
        self.config.transpiler_options.setdefault("routing_method", "basic")
        self.config.transpiler_options.setdefault("translation_method", "translator")
        try:
            self._passmanager = generate_preset_pass_manager(
                **self.config.transpiler_options,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to set up pass manager with options "
                f"{self.config.transpiler_options}: {e}"
            ) from e

        self._sampler = self._build_sampler()

    def _build_sampler(self) -> SamplerV2:
        """Builds and configures a SamplerV2 instance for the backend.

        Returns:
            A configured SamplerV2 instance.

        Raises:
            RuntimeError: If SamplerV2 initialization fails.
        """
        try:
            sampler = SamplerV2(mode=self._backend)

            if self.config.sampler_options is not {}:
                sampler.options.update(**self.config.sampler_options)

            return sampler
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SamplerV2 with backend "
                f"'{getattr(self._backend, 'name', repr(self._backend))}': {e}"
            ) from e

    def execute(
        self, circuit: QuantumCircuit, simulator_seed: int | None = None
    ) -> ExecutionResult:
        """Executes a quantum circuit using IBM Runtime [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2).

        Uses the pass manager and sampler that were constructed during
        ``__init__``. Internally performs:

        ```python
        # 1) Transpile via the pre-built pass manager
        transpiled = self._passmanager.run(circuit)

        # 2) Run via the pre-built sampler
        shots = self.config.run_options.get("shots", 1024)
        job = self._sampler.run([transpiled], shots=shots)
        result = job.result()
        counts = result[0].data.get_counts()
        ```

        Args:
            circuit: The quantum circuit to execute.
            simulator_seed: If not None, overrides the sampler's
                simulator.seed_simulator for this execution.

        Returns:
            An ExecutionResult containing counts, the raw result, and metadata.

        Raises:
            RuntimeError: If transpilation or execution fails.
        """
        # 0) Apply simulator seed override
        original_seed = self._get_configured_seed()
        if simulator_seed is not None:
            self._sampler.options.simulator.seed_simulator = simulator_seed

        try:
            # 1) Transpilation
            if self.config.skip_transpilation:
                transpiled = circuit
                transpile_time = 0.0
            else:
                try:
                    start_transpile = time.perf_counter()
                    transpiled = self._passmanager.run(circuit)
                    transpile_time = time.perf_counter() - start_transpile
                except Exception as e:
                    raise RuntimeError(f"Failed to transpile circuit: {e}") from e

            # 2) Execution via SamplerV2
            shots = self.config.run_options.get("shots", 1024)
            start_execution = time.perf_counter()
            try:
                job = self._sampler.run([transpiled], shots=shots)
                result = job.result()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute circuit on backend "
                    f"'{getattr(self._backend, 'name', 'unknown')}': {e}"
                ) from e
            execution_time = time.perf_counter() - start_execution

            # 3) Extract counts
            counts = self._extract_counts(result)

            return ExecutionResult(
                counts=counts,
                result=result,
                metadata={
                    "transpiled_depth": transpiled.depth(),
                    "transpiled_gate_count": transpiled.size(),
                    "transpile_time_s": transpile_time,
                    "execution_time_s": execution_time,
                    "total_time_s": transpile_time + execution_time,
                },
            )
        finally:
            # Restore original seed state
            if simulator_seed is not None:
                self._restore_seed(original_seed)

    def _get_configured_seed(self) -> int | None:
        """Reads the originally configured simulator seed.

        Returns:
            The configured seed value, or None if not set.
        """
        sim_opts = self.config.sampler_options.get("simulator", {})
        return sim_opts.get("seed_simulator", None)

    def _restore_seed(self, original_seed: int | None) -> None:
        """Restores the sampler's simulator seed to its original value.

        Args:
            original_seed: The seed value to restore, or None if no seed
                was originally configured.
        """
        if original_seed is not None:
            self._sampler.options.simulator.seed_simulator = original_seed

    @staticmethod
    def _extract_counts(result) -> dict[str, int]:
        """Extracts counts from a SamplerV2 PrimitiveResult.

        Args:
            result: The raw PrimitiveResult returned by SamplerV2.

        Returns:
            A dictionary mapping bitstrings to their observed counts.

        Raises:
            RuntimeError: If counts cannot be extracted from
                the result.
        """
        pub_result = result[0]
        data = pub_result.data

        for field_name in data.keys():
            attribute = getattr(data, field_name)
            if hasattr(attribute, "get_counts"):
                return attribute.get_counts()

        raise RuntimeError(
            "Could not extract counts from result. "
            "Ensure the circuit contains measurements."
        )
