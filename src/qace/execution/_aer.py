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
"""Aer-based circuit execution module.

Provides executor implementations that leverage the Qiskit Aer simulator
for running quantum circuits. Includes support for both direct AerSimulator
usage and simulation of [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2)-compatible backends.
"""

from __future__ import annotations

import os
import time, copy

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.providers.backend import BackendV2

from qace.execution._base import CircuitExecutor, ExecutionResult, ExecutionConfig


class AerExecutor(CircuitExecutor):
    """A circuit executor that uses the Qiskit AerSimulator backend.

    This executor wraps the Qiskit Aer simulation pipeline. The setup is
    split between construction time and execution time:

    During ``__init__``:

    ```python
    # Step 1: Backend construction
    backend = AerSimulator(**config.backend_options)

    # Step 2: Pass manager construction
    pass_manager = generate_preset_pass_manager(
        backend=backend, **config.transpiler_options
    )
    ```

    During each ``execute()`` call:

    ```python
    # Step 3: Transpilation
    transpiled = pass_manager.run(circuit)

    # Step 4: Execution
    job = backend.run(transpiled, **config.run_options)
    result = job.result()
    counts = result.get_counts()
    ```

    ## Understanding `ExecutionConfig` fields for `AerExecutor`

    **`backend_options`** — passed to the `AerSimulator(...)` constructor:

    These configure the simulator instance itself (simulation method,
    device, precision, noise model, etc.). They are set once at
    construction time and persist across all `execute()` calls.

    Common examples:

    ```python
    backend_options = {
        "method": "statevector",       # simulation method (see AerSimulator.available_methods())
        "device": "CPU",               # or "GPU" (see AerSimulator.available_devices())
        "precision": "double",         # or "single" (halves memory usage)
        "noise_model": noise_model,    # a qiskit_aer.noise.NoiseModel instance
        "seed_simulator": 42,          # global seed for reproducible results
        "max_parallel_threads": 0,     # 0 = use all available CPU cores
    }
    ```

    Full list: see `AerSimulator` constructor parameters and the
    "Additional Backend Options" section in the [Qiskit Aer documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).

    **`run_options`** — passed to `AerSimulator.run(circuit, ...)` as kwargs:

    These are per-execution options that control how the compiled circuit
    is actually run (e.g. number of shots, memory output). They can
    override backend-level defaults for that specific run.

    Common examples:

    ```python
    run_options = {
        "shots": 1024,            # number of measurement samples
        "seed_simulator": 42,     # per-run seed (overrides backend-level seed)
        "memory": True,           # return per-shot measurement results
    }
    ```

    Full list: see `AerSimulator.run()` [documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html). Any kwarg
    accepted by `run()` can be specified here.

    **`transpiler_options`** — passed to
    `qiskit.generate_preset_pass_manager(...)` as kwargs:

    These control how the quantum circuit is compiled/optimized before
    execution. The `backend` kwarg is automatically set to the
    constructed `AerSimulator` instance; do NOT include it manually.
    The pass manager is built once during ``__init__`` and reused for
    every ``execute()`` call.

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

    Full list: see the [documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.generate_preset_pass_manager)
    of `qiskit.generate_preset_pass_manager()`.

    ## Example usage

    ```python
    from qace.execution import AerExecutor, ExecutionConfig

    config = ExecutionConfig(
        backend_options={"method": "statevector", "seed_simulator": 42},
        run_options={"shots": 2048},
        transpiler_options={"optimization_level": 1, "seed_transpiler": 42},
    )
    executor = AerExecutor(config)
    result = executor.execute(circuit)
    print(result.counts)
    ```
    """

    def __init__(self, config: ExecutionConfig | None = None):
        """Initializes the executor with the given configuration.

        Constructs both the ``AerSimulator`` backend and the preset pass
        manager at construction time so they can be reused across multiple
        ``execute()`` calls:

        ```python
        self._backend = AerSimulator(**config.backend_options)
        self._passmanager = generate_preset_pass_manager(
            backend=self._backend, **config.transpiler_options
        )
        ```

        The pass manager is configured with the following defaults if not
        explicitly specified in ``config.transpiler_options``:

        - ``optimization_level``: 0
        - ``layout_method``: "trivial"
        - ``routing_method``: "basic"
        - ``translation_method``: "translator"

        Args:
            config: Optional execution configuration. Defaults to a
                standard ExecutionConfig if not provided.

        Raises:
            RuntimeError: If the pass manager cannot be set up with the
                provided transpiler options.
        """
        super().__init__()
        self.config = copy.deepcopy(config) or ExecutionConfig()
        self._backend = self._build_backend()

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

    def _build_backend(self) -> AerSimulator:
        """Builds and returns an AerSimulator instance from the configuration.

        Equivalent to:

        ```python
        AerSimulator(**self.config.backend_options)
        ```

        Returns:
            An initialized AerSimulator with the configured backend options.

        Raises:
            RuntimeError: If the AerSimulator fails to initialize.
        """
        try:
            return AerSimulator(**self.config.backend_options)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize AerSimulator with options "
                f"{self.config.backend_options}: {e}"
            ) from e

    def execute(
        self, circuit: QuantumCircuit, simulator_seed: int | None = None
    ) -> ExecutionResult:
        """Transpiles and executes a quantum circuit on the Aer backend.

        Uses the pass manager and backend that were constructed during
        ``__init__``. Internally performs:

        ```python
        # 1) Transpile via the pre-built pass manager
        transpiled = self._passmanager.run(circuit)

        # 2) Run on the pre-built backend
        job = self._backend.run(transpiled, **self.config.run_options)
        result = job.result()
        counts = result.get_counts()
        ```

        Args:
            circuit: The quantum circuit to execute.
            simulator_seed: If not None, overrides ``seed_simulator`` in
                ``run_options`` for this single execution only.

        Returns:
            An ExecutionResult containing measurement counts, the raw result
            object, and metadata describing transpiled circuit depth, gate
            count, and timing information for transpilation and execution.

        Raises:
            RuntimeError: If transpilation of the circuit fails or if
                execution on the backend fails.
        """
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
                raise RuntimeError(
                    f"Failed to transpile circuit with options "
                    f"{self.config.transpiler_options}: {e}"
                ) from e

        # 2) Execution
        run_options = self.config.run_options.copy()
        if simulator_seed is not None:
            run_options["seed_simulator"] = simulator_seed

        start_execution = time.perf_counter()
        try:
            job = self._backend.run(transpiled, **run_options)
            result = job.result()
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute circuit on backend "
                f"'{getattr(self._backend, 'name', 'unknown')}': {e}"
            ) from e
        execution_time = time.perf_counter() - start_execution

        # 3) Setup result
        counts = result.get_counts()
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

    """Defines a default `AerExecutor`, which utilizes the GPU if possible, draws a random seed directly from 32 bytes returned by the `os.urandom` interface, and delivers one shot. Also skips transpilation.

            Returns:
                The `AerExecutor` described.
            """

    @staticmethod
    def default(seeded: bool = False) -> AerExecutor:
        """Creates an AerExecutor with default configuration.

        Selects the GPU device if available, otherwise falls back to the
        CPU, and configures the executor for single-shot execution with
        transpilation skipped.

        Args:
            seeded: If True, seeds the simulator with a random value derived
                from the operating system's entropy source, so that repeated
                measurements of the same circuit always yield identical
                results. If False (the default), the simulator is left
                unseeded, so repeated measurements of the same circuit may
                yield different results.

        Returns:
            An AerExecutor configured with the default execution settings.
        """
        device = "GPU" if "GPU" in AerSimulator().available_devices() else "CPU"
        b_options = {"device": device}
        if seeded:
            b_options["seed_simulator"] = int.from_bytes(os.urandom(4))
        executor = AerExecutor(
            ExecutionConfig(
                backend_options=b_options,
                run_options={"shots": 1},
                skip_transpilation=True,
            )
        )
        return executor


class AerBackendV2Executor(AerExecutor):
    """An Aer executor that simulates a specific [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) device.

    Constructs an AerSimulator that mimics an IBM Quantum backend (or any
    [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2)), including its noise model and coupling map. The setup is
    split between construction time and execution time:

    During ``__init__``:

    ```python
    # Step 1: Backend construction
    real_backend = config.backend_options["backend"]  # a BackendV2 instance
    other_opts = {k: v for k, v in config.backend_options.items() if k != "backend"}
    backend = AerSimulator.from_backend(real_backend, **other_opts)

    # Step 2: Pass manager construction
    pass_manager = generate_preset_pass_manager(
        backend=backend, **config.transpiler_options
    )
    ```

    During each ``execute()`` call:

    ```python
    # Step 3: Transpilation
    transpiled = pass_manager.run(circuit)

    # Step 4: Execution
    job = backend.run(transpiled, **config.run_options)
    result = job.result()
    counts = result.get_counts()
    ```

    Understanding ``ExecutionConfig`` fields for ``AerBackendV2Executor``:

    ``backend_options``:
        Must contain a ``"backend"`` key with a ``BackendV2`` instance.
        All other entries are passed as extra kwargs to
        ``AerSimulator.from_backend()``.

        ```python
        backend_options = {
            "backend": FakeManilaV2(),   # REQUIRED: BackendV2 to simulate
            "seed_simulator": 42,        # optional: additional AerSimulator options
        }
        ```

    ``run_options``:
        Same as ``AerExecutor`` — passed to ``backend.run(circuit, ...)``.

        ```python
        run_options = {
            "shots": 4096,
            "seed_simulator": 42,
            "memory": True,
        }
        ```

    ``transpiler_options``:
        Same as ``AerExecutor`` — passed to
        ``qiskit.generate_preset_pass_manager(...)``. The following
        defaults are applied if not explicitly provided:

        - ``optimization_level``: 0
        - ``layout_method``: "trivial"
        - ``routing_method``: "basic"
        - ``translation_method``: "translator"

        ```python
        transpiler_options = {
            "optimization_level": 2,
            "seed_transpiler": 42,
        }
        ```

    Example usage:

    ```python
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qace.execution import AerBackendV2Executor, ExecutionConfig

    config = ExecutionConfig(
        backend_options={"backend": FakeManilaV2(), "seed_simulator": 42},
        run_options={"shots": 4096},
        transpiler_options={"optimization_level": 2},
    )
    executor = AerBackendV2Executor(config)
    result = executor.execute(circuit)
    print(result.counts)
    ```
    """

    def __init__(self, config: ExecutionConfig | None = None):
        """Initializes the executor with a [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) target from the config.

        Internally constructs the simulator and pass manager during
        construction via the parent class:

        ```python
        AerSimulator.from_backend(
            config.backend_options["backend"],
            **{k: v for k, v in config.backend_options.items() if k != "backend"}
        )
        ```

        The pass manager is configured with the following defaults if not
        explicitly specified in ``config.transpiler_options``:

        - ``optimization_level``: 0
        - ``layout_method``: "trivial"
        - ``routing_method``: "basic"
        - ``translation_method``: "translator"

        Args:
            config: Optional execution configuration. Must include a
                'backend' entry in backend_options.

        Raises:
            RuntimeError: If 'backend' is missing from backend_options
                or is not a [BackendV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.BackendV2) instance.
        """
        executor_config = config or ExecutionConfig()

        backend = executor_config.backend_options.get("backend", None)

        if backend is None:
            raise RuntimeError(
                "backend_options must contain a 'backend' key with a BackendV2 to simulate."
            )
        elif not isinstance(backend, BackendV2):
            raise RuntimeError(
                f"backend_options['backend'] must be a BackendV2 instance, "
                f"got {type(backend).__name__}."
            )

        super().__init__(executor_config)

    def _build_backend(self) -> AerSimulator:
        """Builds an AerSimulator from the provided BackendV2 instance.

        Equivalent to:

        ```python
        AerSimulator.from_backend(backend, **other_options)
        ```

        Returns:
            An AerSimulator configured to mimic the target BackendV2.

        Raises:
            RuntimeError: If creating the AerSimulator from the
                backend fails.
        """
        options = {
            k: v for k, v in self.config.backend_options.items() if k != "backend"
        }
        backend = self.config.backend_options["backend"]
        try:
            return AerSimulator.from_backend(backend, **options)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create AerSimulator from backend "
                f"'{getattr(backend, 'name', repr(backend))}': {e}"
            ) from e
