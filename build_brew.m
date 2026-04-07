function build_brew()
%BUILD_BREW Generate and compile the brew MEX gateway + MATLAB wrappers.
%   1. Runs generate_mex.py to regenerate brew_mex.cpp and +BREW wrappers
%      from @mex annotations in the C++ headers.
%   2. Auto-discovers C++ source files needed for compilation.
%   3. Compiles the MEX binary.
%
%   Requires Python 3.6+ on the system PATH and a C++20 compiler.
%   Eigen is automatically downloaded on first build to deps/eigen.
%
%   Usage:
%       build_brew

    here = fileparts(mfilename('fullpath'));
    brew_root = fullfile(here, 'brew');
    gen_dir = fullfile(here, 'generator_files');

    % f = waitbar(0,'Please wait...');

    % =================================================================
    % Step 1: Run Python generator
    % =================================================================
    fprintf('--- Step 1: Generating brew_mex.cpp and MATLAB wrappers ---\n');

    gen_script = fullfile(gen_dir, 'generate_mex.py');
    if ~isfile(gen_script)
        error('brew:build', 'generate_mex.py not found in %s', gen_dir);
    end

    cmd = sprintf('python "%s" --output "%s" --include-root "%s" --matlab-dir "%s"', ...
        gen_script, ...
        fullfile(gen_dir, 'brew_mex.cpp'), ...
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
    sources = {fullfile(gen_dir, 'brew_mex.cpp')};

    % Auto-discover .cpp files from relevant subdirectories
    src_base = fullfile(brew_root, 'src', 'brew');
    scan_dirs = {'dynamics', 'filters', 'fusion', 'clustering', 'models', ...
                 'multi_target', 'assignment', 'metrics', 'template_matching'};

    % Files to skip (plot, I/O with external deps)
    skip_patterns = {'point_cloud_io'};

    num_files = numel(scan_dirs);

    for i = 1:num_files
        % waitbar(i / num_files,f,'Collecting Source Files');
        d = fullfile(src_base, scan_dirs{i});
        if isfolder(d)
            cpps = dir(fullfile(d, '*.cpp'));
            for j = 1:numel(cpps)
                skip = false;
                for k = 1:numel(skip_patterns)
                    if contains(cpps(j).name, skip_patterns{k})
                        skip = true;
                        break;
                    end
                end
                if ~skip
                    sources{end+1} = fullfile(d, cpps(j).name); 
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

    % waitbar(1,f,'Compiling MEX');

    % =================================================================
    % Step 3: Compile MEX
    % =================================================================
    fprintf('\n--- Step 3: Compiling brew_mex ---\n');
    fprintf('  Brew include: %s\n', brew_include);
    fprintf('  Eigen include: %s\n', eigen_include);
    fprintf('  This could take a few minutes.\n');

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

    % close(f)

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
