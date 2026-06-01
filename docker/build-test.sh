#!/usr/bin/env bash
# Verify the brew C++ library with no MATLAB:
#   1. Each model package compiles standalone (hardware target, BREW_MODELS=<m>).
#   2. The full library + GoogleTest suite builds and passes.
#   3. The hardware cross-compile-friendly target builds (exceptions off).
#
# Usage (inside the container): build-test.sh [model ...]
#   no args  -> all packages
set -euo pipefail
cd /work/brew

ALL_MODELS=(gaussian ggiw ggiw_orientation iggiw template_pose
            trajectory_gaussian trajectory_ggiw trajectory_ggiw_orientation
            trajectory_iggiw trajectory_template_pose)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "==== [1/3] Per-package standalone library builds (hardware target) ===="
for M in "${MODELS[@]}"; do
    echo "---- package: $M ----"
    cmake -S . -B "build-$M" -G Ninja -DBREW_TARGET=hardware -DBREW_MODELS="$M" >/dev/null
    cmake --build "build-$M" --target "brew_pkg_$M"
done

echo "==== [2/3] Full desktop build + GoogleTest suite ===="
cmake -S . -B build-all -G Ninja \
    -DBREW_TARGET=desktop -DBREW_ENABLE_PLOTTING=OFF \
    -DBREW_BUILD_EXAMPLES=OFF -DBREW_BUILD_TESTS=ON >/dev/null
cmake --build build-all
ctest --test-dir build-all --output-on-failure

echo "==== [3/3] Hardware target (exceptions off), gaussian only ===="
cmake -S . -B build-hw -G Ninja -DBREW_TARGET=hardware -DBREW_MODELS=gaussian >/dev/null
cmake --build build-hw

echo ""
echo "ALL CHECKS PASSED for: ${MODELS[*]}"
