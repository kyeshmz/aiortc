Contributing
============

Thanks for taking the time to contribute to ``aiortc``!

Code of Conduct
---------------

This project and everyone participating in it is governed by the `Code of
Conduct`_. By participating, you are expected to uphold this code. Please
report inappropriate behavior to jeremy DOT laine AT m4x DOT org.

.. _Code of Conduct: https://github.com/aiortc/aiortc/blob/main/CODE_OF_CONDUCT.md

Contributions
-------------

Bug reports, patches and suggestions are welcome!

Please open an issue_ or send a `pull request`_.

Feedback about the examples or documentation are especially valuable as they
make ``aiortc`` accessible to a wider audience.

Code contributions *must* come with full unit test coverage. WebRTC is a
complex protocol stack and ensuring correct behaviour now and in the future
requires a proper investment in automated testing.

.. _issue: https://github.com/aiortc/aiortc/issues/new
.. _pull request: https://github.com/aiortc/aiortc/compare/

Measure full test coverage
--------------------------

Run every test with the default ``unittest discover`` command. Do not restrict
discovery to selected modules when you measure package-wide coverage. CI keeps
the Python 3.11 through 3.15 compatibility jobs and each job's Codecov XML upload.
The exact coverage gate combines raw branch data from the Python 3.11 and 3.14
jobs in the same workflow run.

Use two environments with real Python 3.11 and 3.14 interpreters. Install the
development dependencies in each environment. Run these Bash commands from the
repository root without changing the checkout between runs. Replace the two
interpreter paths with your environment paths.

.. code-block:: bash

   PY311=/absolute/path/to/python3.11
   PY314=/absolute/path/to/python3.14
   export PYTHONPATH="$PWD/src"
   "$PY311" -m pip install '.[dev]'
   "$PY314" -m pip install '.[dev]'
   "$PY311" - <<'PY'
   import sys
   from pathlib import Path
   import aiortc
   assert sys.version_info[:2] == (3, 11)
   assert Path(aiortc.__file__).resolve() == Path("src/aiortc/__init__.py").resolve()
   PY
   COVERAGE_FILE=.coverage.3.11 "$PY311" -m coverage run -m unittest discover -v
   "$PY314" - <<'PY'
   import sys
   from pathlib import Path
   import aiortc
   assert sys.version_info[:2] == (3, 14)
   assert Path(aiortc.__file__).resolve() == Path("src/aiortc/__init__.py").resolve()
   PY
   COVERAGE_FILE=.coverage.3.14 "$PY314" -m coverage run -m unittest discover -v

Keep ``PYTHONPATH`` set for the remaining commands. It selects the checked-out
``src/aiortc`` package instead of a copy in ``site-packages``. Both raw inputs
must contain branch data for exactly the same package source paths. Validate
the inputs before combining them.

.. code-block:: bash

   export COVERAGE_FILE=.coverage.combined
   "$PY314" - <<'PY'
   from pathlib import Path
   import aiortc
   from coverage import CoverageData

   source = Path("src/aiortc").resolve()
   assert Path(aiortc.__file__).resolve() == source / "__init__.py"
   expected = {str(path) for path in source.rglob("*.py")}
   for version in ("3.11", "3.14"):
       path = Path(f".coverage.{version}")
       assert path.is_file(), f"Missing raw coverage input: {path}"
       data = CoverageData(basename=str(path))
       data.read()
       assert data.has_arcs(), f"Statement-only coverage input: {path}"
       assert set(data.measured_files()) == expected, f"Source inventory mismatch: {path}"
   PY
   "$PY314" -m coverage combine --keep .coverage.3.11 .coverage.3.14
   "$PY314" -m coverage json -o coverage.json
   "$PY314" - <<'PY'
   import json
   from pathlib import Path
   import aiortc

   assert Path(aiortc.__file__).resolve() == Path("src/aiortc/__init__.py").resolve()
   report = json.loads(Path("coverage.json").read_text())
   expected = {path.as_posix() for path in Path("src/aiortc").rglob("*.py")}
   assert report["meta"]["branch_coverage"], "Combined data has no branches"
   assert set(report["files"]) == expected, "Combined source inventory mismatch"
   for filename, result in report["files"].items():
       assert result["summary"]["missing_lines"] == 0, f"Missing statements: {filename}"
       assert result["summary"]["missing_branches"] == 0, f"Missing branches: {filename}"
   totals = report["totals"]
   assert totals["num_statements"] > 0 and totals["num_branches"] > 0
   assert totals["missing_lines"] == 0 and totals["missing_branches"] == 0, totals
   print(f"Exact coverage verified for {len(expected)} package source files.")
   PY
   "$PY314" -m coverage report --show-missing --fail-under=100

``combine --keep`` preserves both raw inputs. The combined database uses
``.coverage.combined`` so reporting does not implicitly recombine or delete
the ``.coverage.3.11`` and ``.coverage.3.14`` inputs. The JSON checks require zero
missing statements and branches for every package source file, including files
with no executed code. The inventory comes from ``src/aiortc/**/*.py`` and
currently contains 33 files. A rounded display of ``100%`` is not sufficient.
CI also checks this inventory before combining, so data from installed copies
or a different workspace path cannot satisfy the gate.

The union is required because Unix-socket cleanup differs between Python
versions. Python 3.13 added automatic Unix-server socket cleanup. The tests
exercise both the older cleanup path and the newer path on real runtimes.
An individual runtime must pass every test, but it need not reach 100% alone.
Run the suites sequentially because signaling tests share a port and socket
path. Do not combine data from different revisions or simulate Python versions.

The existing coverage policy excludes 11 lines marked ``pragma: no cover``.
This gate does not change those exclusions or omit any package source files.
Coverage measures Python statements and branches, not native code in PyAV,
FFmpeg, cryptography, or other dependencies. If a supported-input invariant
makes code unreachable, prove the invariant before removing the dead code.
Do not lower the denominator by deleting tests, narrowing discovery, adding
exclusions or skips, or removing reachable behavior.

Questions
---------

GitHub issues aren't a good medium for handling questions. There are better
places to ask questions, for example Stack Overflow.

If you want to ask a question anyway, please make sure that:

- it's a question about ``aiortc`` and not about :mod:`asyncio`;
- it isn't answered by the documentation;
- it wasn't asked already.

A good question can be written as a suggestion to improve the documentation.
