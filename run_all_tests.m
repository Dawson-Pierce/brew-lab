function results = run_all_tests(varargin)
%RUN_ALL_TESTS Run every MATLAB test script in tests/ one by one.
%   Executes each test script (tests/test*.m and tests/Test*.m) in an isolated
%   workspace, catching errors so a failure in one test does not stop the rest,
%   closing figures between runs, and printing a pass/fail summary at the end.
%   A test "passes" if its script runs to completion without throwing.
%
%   Requires the MEX wrapper to be built first (run build_brew).
%
%   Usage:
%       run_all_tests                      % run all test scripts in tests/
%       run_all_tests('test_PHD_GGIW.m')   % run only the named script(s)
%       results = run_all_tests(...)       % also return a struct array of
%                                          % results: name, passed, time, error
%
%   Run this from the project root (the folder containing +BREW and build_brew.m).

    root    = fileparts(mfilename('fullpath'));   % project root (has +BREW, +utils)
    testdir = fullfile(root, 'tests');

    if ~isfolder(testdir)
        error('run_all_tests:noTestDir', 'Test folder not found: %s', testdir);
    end

    % Make the +BREW / +utils packages resolvable and run from the project root
    % so the scripts' relative output paths (e.g. tests/output/*.gif) work.
    orig_path = path;
    orig_dir  = pwd;
    restore   = onCleanup(@() local_restore(orig_dir, orig_path)); %#ok<NASGU>
    addpath(root, testdir);   % +BREW/+utils resolvable; test scripts callable by name
    cd(root);                 % scripts write outputs relative to the project root

    % --- Collect the test scripts -------------------------------------------
    if nargin > 0
        names = varargin;                          % explicit subset
    else
        files = dir(fullfile(testdir, '*.m'));
        names = {};
        for k = 1:numel(files)
            if startsWith(lower(files(k).name), 'test')   % test_*.m and Test*.m
                names{end+1} = files(k).name;             %#ok<AGROW>
            end
        end
        names = sort(names);
    end

    if isempty(names)
        fprintf('No test scripts found in %s\n', testdir);
        results = struct('name', {}, 'passed', {}, 'time', {}, 'error', {});
        return;
    end

    fprintf('Running %d test script(s) from %s\n', numel(names), testdir);
    fprintf('%s\n', repmat('=', 1, 72));

    results = struct('name', {}, 'passed', {}, 'time', {}, 'error', {});
    for k = 1:numel(names)
        name = names{k};
        scriptPath = fullfile(testdir, name);
        fprintf('\n[%d/%d] %s\n', k, numel(names), name);

        if ~isfile(scriptPath)
            fprintf('    SKIP - not found\n');
            results(end+1) = struct('name', name, 'passed', false, ...
                'time', 0, 'error', 'file not found'); %#ok<AGROW>
            continue;
        end

        t0 = tic;
        passed = true;
        errmsg = '';
        try
            run_isolated(name);
        catch ME
            passed = false;
            errmsg = getReport(ME, 'extended', 'hyperlinks', 'off');
        end
        elapsed = toc(t0);
        close all force;          % free figures/animations between tests
        drawnow;

        if passed
            fprintf('    PASS (%.1f s)\n', elapsed);
        else
            fprintf('    FAIL (%.1f s)\n', elapsed);
            % Indented error report for readability.
            fprintf('      %s\n', strrep(strtrim(errmsg), newline, [newline '      ']));
        end
        results(end+1) = struct('name', name, 'passed', passed, ...
            'time', elapsed, 'error', errmsg); %#ok<AGROW>
    end

    % --- Summary ------------------------------------------------------------
    npass = sum([results.passed]);
    nfail = numel(results) - npass;
    fprintf('\n%s\n', repmat('=', 1, 72));
    fprintf('SUMMARY: %d passed, %d failed of %d\n\n', npass, nfail, numel(results));
    for k = 1:numel(results)
        if results(k).passed, status = 'PASS'; else, status = 'FAIL'; end
        fprintf('  %-4s  %-38s  %7.1f s\n', status, results(k).name, results(k).time);
    end
    if nfail > 0
        fprintf('\n%d test(s) FAILED:\n', nfail);
        for k = 1:numel(results)
            if ~results(k).passed
                fprintf('  - %s\n', results(k).name);
            end
        end
    end

    if nargout == 0
        clear results;
    end
end

% -------------------------------------------------------------------------
function run_isolated(name)
%RUN_ISOLATED Execute a test script (by name, found on the path) in this
%   function's fresh workspace so test variables - and the BREW handle objects
%   they create, which free their C++ objects on destruction - do not leak
%   between tests. The script is run WITHOUT changing directory (unlike
%   run(fullpath)), so its project-root-relative output paths (e.g.
%   tests/output/*.gif) resolve correctly.
    stem = erase(name, '.m');
    eval(stem);
end

function local_restore(orig_dir, orig_path)
    cd(orig_dir);
    path(orig_path);
end
