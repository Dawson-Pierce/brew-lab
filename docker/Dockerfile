# Build/verify the brew C++ library and its tests in isolation, with NO MATLAB.
#
# This image compiles the pure C++ library + GoogleTest suite so each model
# package can be confirmed to build and pass standalone (the "is each thing
# loadable" check), and exercises the hardware cross-compile-friendly target.
#
# Build context is the repository root (brew-lab/):
#   docker build -f docker/Dockerfile -t brew-verify .
#   docker run --rm brew-verify
#
# GoogleTest / EigenRand are fetched by the brew CMake via FetchContent, so the
# container needs network on first run (or a pre-seeded dependencies/ cache).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
# Only the C++ library + its verification harness are needed (no MATLAB layer).
COPY brew /work/brew
COPY docker/build-test.sh /work/build-test.sh
RUN chmod +x /work/build-test.sh

# Default: build + test every package standalone, the full suite, and the
# hardware target. Override the model list with `docker run ... <models...>`.
ENTRYPOINT ["/work/build-test.sh"]
