# brew C++ verification (Docker)

Compiles the **C++ library + GoogleTest suite** with no MATLAB, to confirm each
model package builds and passes standalone. This is the "is each thing loadable"
check and exercises the `hardware` cross-compile target.

## Run

From the repository root (`brew-lab/`):

```bash
docker build -f docker/Dockerfile -t brew-verify .
docker run --rm brew-verify                 # all packages + full suite + hardware
docker run --rm brew-verify gaussian ggiw   # only these packages
```

The container performs three checks (see `docker/build-test.sh`):

1. **Per-package standalone build** — `cmake -DBREW_TARGET=hardware -DBREW_MODELS=<m>`
   then build only `brew_pkg_<m>`. Confirms each package compiles on its own.
2. **Full desktop build + tests** — all packages, plotting off, then `ctest`.
3. **Hardware target** — `BREW_TARGET=hardware` (exceptions off) for `gaussian`,
   confirming the embedded-friendly path compiles.

GoogleTest / EigenRand are fetched via CMake `FetchContent` on first run, so the
build needs network (or a pre-seeded `brew/dependencies/`).

## Selecting objects to compile

`BREW_MODELS` is a semicolon list of the packages to build (default: all). For a
hardware/embedded target you typically compile only what you need, e.g.:

```bash
cmake -S brew -B build -DBREW_TARGET=hardware -DBREW_MODELS="gaussian;trajectory_gaussian"
```

The MATLAB MEX build (`build_brew.m`) always compiles every model regardless of
`BREW_MODELS`, so the MATLAB side stays comprehensive.
