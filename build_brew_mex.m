function build_brew_mex()
%BUILD_BREW_MEX Compile the brew_mex MEX gateway.
%   Compiles brew_mex.cpp along with all required brew C++ source files.
%   Requires a C++20-capable compiler (MSVC 2019+, GCC 10+, Clang 12+).
%
%   Eigen is automatically downloaded on first build to deps/eigen.
%   To use a custom Eigen, set the EIGEN_DIR environment variable.
%
%   Usage:
%       build_brew_mex          % builds with default settings

    here = fileparts(mfilename('fullpath'));
    brew_root = fullfile(here, 'brew');

    % --- Include paths ---
    brew_include = fullfile(brew_root, 'include');

    % --- Find or fetch Eigen ---
    eigen_include = find_eigen(here);
    if isempty(eigen_include)
        eigen_include = fetch_eigen(here);
    end

    % --- Source files ---
    src_base = fullfile(brew_root, 'src', 'brew');
    sources = {
        fullfile(here, 'brew_mex.cpp')
        fullfile(src_base, 'filters', 'ekf.cpp')
        fullfile(src_base, 'filters', 'ggiw_ekf.cpp')
        fullfile(src_base, 'filters', 'trajectory_gaussian_ekf.cpp')
        fullfile(src_base, 'filters', 'trajectory_ggiw_ekf.cpp')
        fullfile(src_base, 'fusion', 'merge.cpp')
        fullfile(src_base, 'clustering', 'dbscan.cpp')
        fullfile(src_base, 'dynamics', 'linear_dynamics.cpp')
    };

    % Check that all source files exist
    for i = 1:numel(sources)
        if ~isfile(sources{i})
            error('brew:build', 'Source file not found: %s', sources{i});
        end
    end

    % --- Compiler flags ---
    include_flags = {
        ['-I' brew_include]
        ['-I' eigen_include]
    };

    if ispc
        cpp_flag = 'COMPFLAGS=$COMPFLAGS /std:c++20 /EHsc /bigobj /D_USE_MATH_DEFINES';
    else
        cpp_flag = 'CXXFLAGS=$CXXFLAGS -std=c++20';
    end

    % --- Build ---
    fprintf('Building brew_mex...\n');
    fprintf('  Brew include: %s\n', brew_include);
    fprintf('  Eigen include: %s\n', eigen_include);
    fprintf('  Sources: %d files\n', numel(sources));

    args = [{'-R2018a'} sources(:)' include_flags(:)' {cpp_flag} ...
            {'-output', fullfile(here, 'brew_mex')}];

    mex(args{:});

    fprintf('brew_mex built successfully.\n');
end

function eigen_path = find_eigen(here)
%FIND_EIGEN Search for an existing Eigen installation.
%   Checks EIGEN_DIR env var, then local deps folder, then system paths.
    eigen_path = getenv('EIGEN_DIR');
    if ~isempty(eigen_path) && isfolder(eigen_path)
        return;
    end

    candidates = {
        fullfile(here, 'deps', 'eigen')
        fullfile(here, 'brew', 'build', '_deps', 'eigen-src')
        '/usr/include/eigen3'
        '/usr/local/include/eigen3'
    };
    for i = 1:numel(candidates)
        if isfolder(fullfile(candidates{i}, 'Eigen'))
            eigen_path = candidates{i};
            return;
        end
    end

    eigen_path = '';
end

function eigen_path = fetch_eigen(here)
%FETCH_EIGEN Download Eigen headers from GitLab release archive.
    eigen_version = '3.4.0';
    eigen_dir = fullfile(here, 'deps', 'eigen');

    url = sprintf( ...
        'https://gitlab.com/libeigen/eigen/-/archive/%s/eigen-%s.tar.gz', ...
        eigen_version, eigen_version);

    deps_dir = fullfile(here, 'deps');
    if ~isfolder(deps_dir)
        mkdir(deps_dir);
    end

    archive = fullfile(deps_dir, 'eigen.tar.gz');
    fprintf('Downloading Eigen %s...\n', eigen_version);
    websave(archive, url);

    fprintf('Extracting Eigen...\n');
    untar(archive, deps_dir);
    delete(archive);

    % The archive extracts to eigen-<version>; rename to eigen
    extracted = fullfile(deps_dir, sprintf('eigen-%s', eigen_version));
    if ~isfolder(extracted)
        error('brew:build', 'Eigen extraction failed: %s not found', extracted);
    end
    movefile(extracted, eigen_dir);

    eigen_path = eigen_dir;
    fprintf('Eigen %s installed to %s\n', eigen_version, eigen_path);
end
