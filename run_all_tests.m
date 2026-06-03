function results = run_all_tests(varargin)
%RUN_ALL_TESTS Run every MATLAB test script in tests/ one by one.

    root    = fileparts(mfilename('fullpath'));
    testdir = fullfile(root, 'tests');

    if ~isfolder(testdir)
        error('run_all_tests:noTestDir', 'Test folder not found: %s', testdir);
    end

    orig_path = path;
    orig_dir  = pwd;
    restore   = onCleanup(@() local_restore(orig_dir, orig_path));
    addpath(root, testdir);
    cd(root);

    if nargin > 0
        names = varargin;
    else
        files = dir(fullfile(testdir, '*.m'));
        names = {};
        for k = 1:numel(files)
            if startsWith(lower(files(k).name), 'test')
                names{end+1} = files(k).name;
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
                'time', 0, 'error', 'file not found');
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
        close all force;
        drawnow;

        if passed
            fprintf('    PASS (%.1f s)\n', elapsed);
        else
            fprintf('    FAIL (%.1f s)\n', elapsed);
            fprintf('      %s\n', strrep(strtrim(errmsg), newline, [newline '      ']));
        end
        results(end+1) = struct('name', name, 'passed', passed, ...
            'time', elapsed, 'error', errmsg);
    end

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

function run_isolated(name)
    stem = erase(name, '.m');
    eval(stem);
end

function local_restore(orig_dir, orig_path)
    cd(orig_dir);
    path(orig_path);
end
