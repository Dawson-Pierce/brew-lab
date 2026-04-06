function build_brew_mex()
%BUILD_BREW_MEX Generate and compile the brew_mex MEX gateway.
%   1. Runs generate_mex.py to regenerate brew_mex.cpp and +BREW wrappers
%      from @mex annotations in the C++ headers.
%   2. Auto-discovers C++ source files needed for compilation.
%   3. Compiles the MEX binary.
%
%   Requires Python 3.6+ on the system PATH and a C++20 compiler.
%   Eigen is automatically downloaded on first build to deps/eigen.
%
%   Usage:
%       build_brew_mex          % full rebuild (generate + compile)

    here = fileparts(mfilename('fullpath'));
    brew_root = fullfile(here, 'brew');

    % =================================================================
    % Step 1: Run Python generator
    % =================================================================
    fprintf('--- Step 1: Generating brew_mex.cpp and MATLAB wrappers ---\n');

    gen_script = fullfile(here, 'generate_mex.py');
    if ~isfile(gen_script)
        error('brew:build', 'generate_mex.py not found in %s', here);
    end

    cmd = sprintf('python "%s" --output "%s" --include-root "%s" --matlab-dir "%s"', ...
        gen_script, ...
        fullfile(here, 'brew_mex.cpp'), ...
        fullfile(brew_root, 'include'), ...
        fullfile(here, '+BREW'));

    [status, output] = system(cmd);
    fprintf('%s', output);
    if status ~= 0
        error('brew:build', 'generate_mex.py failed (exit code %d)', status);
    end

    % =================================================================
    % Step 2: Collect source files
    % =================================================================
    fprintf('\n--- Step 2: Discovering source files ---\n');

    % Include paths
    brew_include = fullfile(brew_root, 'include');
    eigen_include = find_eigen(here);
    if isempty(eigen_include)
        eigen_include = fetch_eigen(here);
    end

    % Start with the generated gateway
    sources = {fullfile(here, 'brew_mex.cpp')};

    % Auto-discover .cpp files from relevant subdirectories
    src_base = fullfile(brew_root, 'src', 'brew');
    scan_dirs = {'dynamics', 'filters', 'fusion', 'clustering', 'models', ...
                 'multi_target', 'assignment', 'metrics'};

    for i = 1:numel(scan_dirs)
        d = fullfile(src_base, scan_dirs{i});
        if isfolder(d)
            cpps = dir(fullfile(d, '*.cpp'));
            for j = 1:numel(cpps)
                src = fullfile(d, cpps(j).name);
                % Skip template matching and plot files that may end up here
                if ~contains(cpps(j).name, 'tm_ekf') && ...
                   ~contains(cpps(j).name, 'trajectory_tm')
                    sources{end+1} = src; %#ok<AGROW>
                end
            end
        end
    end

    % Verify all source files exist
    for i = 1:numel(sources)
        if ~isfile(sources{i})
            error('brew:build', 'Source file not found: %s', sources{i});
        end
    end

    fprintf('  Sources: %d files\n', numel(sources));

    % =================================================================
    % Step 3: Compile MEX
    % =================================================================
    fprintf('\n--- Step 3: Compiling brew_mex ---\n');
    fprintf('  Brew include: %s\n', brew_include);
    fprintf('  Eigen include: %s\n', eigen_include);

    include_flags = {
        ['-I' brew_include]
        ['-I' eigen_include]
    };

    if ispc
        cpp_flag = 'COMPFLAGS=$COMPFLAGS /std:c++20 /EHsc /bigobj /D_USE_MATH_DEFINES';
    else
        cpp_flag = 'CXXFLAGS=$CXXFLAGS -std=c++20';
    end

    args = [{'-R2018a'} sources(:)' include_flags(:)' {cpp_flag} ...
            {'-output', fullfile(here, 'brew_mex')}];

    mex(args{:});

    fprintf('\nbrew_mex built successfully.\n');
end


% =================================================================
% Eigen helpers (unchanged from original)
% =================================================================

function eigen_path = find_eigen(here)
%FIND_EIGEN Search for an existing Eigen installation.
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

    extracted = fullfile(deps_dir, sprintf('eigen-%s', eigen_version));
    if ~isfolder(extracted)
        error('brew:build', 'Eigen extraction failed: %s not found', extracted);
    end
    movefile(extracted, eigen_dir);

    eigen_path = eigen_dir;
    fprintf('Eigen %s installed to %s\n', eigen_version, eigen_path);
end
