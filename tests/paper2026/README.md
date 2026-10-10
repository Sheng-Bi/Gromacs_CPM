# Paper tests, inputs and analysis

All publication-specific tests and analysis are organized in this directory.
See [reproduction instructions](../../docs/REPRODUCING.md) for the full workflow.

| Directory | Contents |
|---|---|
| `inputs/slab/` | Shared slab coordinates and topology, included with the code; used by the feature suite and figure scripts. |
| `inputs/slit/` | Slit coordinates and topology, supplied by the companion data archive. |
| `cpm_testsuite/` | Runnable feature-validation cases and assertions. |
| `figures/` | Figure-generation scripts; their saved input data come from the companion archive. |
| `cnt_pair/` | CNT analysis scripts and, in the data archive, completed simulation outputs. |
| `timedep/`, `eis/`, `tf/`, `gpuval3/` | Protocol, model and CPU/GPU validation analyses. |

Additional evidence directories are supplied by the companion data archive under
this same `tests/paper2026/` tree. The code and data archives must be extracted
into the same release directory. No simulation data need to be copied to a
separate benchmark folder.

To run the feature suite from the release root:

```bash
export GMX_BIN="$PWD/build/bin/gmx"
cd tests/paper2026/cpm_testsuite
./run_tests.sh setup
./run_tests.sh --gpu
```

The runner finds the shared slab inputs at `../inputs/slab/` by default.
Set `CPM_SLAB_DIR` only when deliberately using another input directory.
