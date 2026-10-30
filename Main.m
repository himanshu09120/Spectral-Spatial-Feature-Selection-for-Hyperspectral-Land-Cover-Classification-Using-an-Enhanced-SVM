function Main(mode)
%MAIN Portable MATLAB launcher. Main('validate-data') performs no training.
% Set FRSS_PYTHON to an absolute interpreter path if not using project .venv.
% Training is disabled in config/config.json until explicitly enabled there.
    if nargin < 1
        mode = 'run';
    end
    allowed = {'run', 'dry-run', 'validate-data', 'smoke-test'};
    if ~any(strcmp(mode, allowed))
        error('FRSS:Mode', 'Unknown mode. Use run, dry-run, validate-data, or smoke-test.');
    end
    root = fileparts(mfilename('fullpath'));
    originalDirectory = pwd;
    restoreDirectory = onCleanup(@() cd(originalDirectory)); %#ok<NASGU>
    cd(root);
    entryPoint = fullfile(root, 'src', 'main.py');
    if ~isfile(entryPoint)
        error('FRSS:MissingEntry', 'Python entry point not found: %s', entryPoint);
    end
    interpreter = getenv('FRSS_PYTHON');
    if isempty(interpreter)
        if ispc
            candidate = fullfile(root, '.venv', 'Scripts', 'python.exe');
        else
            candidate = fullfile(root, '.venv', 'bin', 'python');
        end
        if isfile(candidate)
            interpreter = candidate;
        else
            candidates = {'python3', 'python'};
            for index = 1:numel(candidates)
                [status, ~] = system([shellQuote(candidates{index}) ' --version']);
                if status == 0
                    interpreter = candidates{index};
                    break;
                end
            end
        end
    end
    if isempty(interpreter)
        error('FRSS:PythonMissing', 'Create .venv or set FRSS_PYTHON. See README.md.');
    end
    command = [shellQuote(interpreter) ' ' shellQuote(entryPoint) ' --config ' ...
               shellQuote(fullfile(root, 'config', 'config.json')) ' --' mode];
    [status, output] = system(command, '-echo');
    if status ~= 0
        error('FRSS:PythonFailed', 'Python exited with status %d.\n%s', status, output);
    end
end

function quoted = shellQuote(value)
    if any(value == char(10)) || any(value == char(13))
        error('FRSS:InvalidPath', 'Paths must not contain newline characters.');
    end
    if ispc
        if any(ismember(value, ['"' '%' '!']))
            error('FRSS:InvalidPath', 'Windows paths must not contain quote, percent or exclamation characters.');
        end
        quoted = ['"' value '"'];
    else
        escape = [char(39) char(34) char(39) char(34) char(39)];
        quoted = [char(39) strrep(value, char(39), escape) char(39)];
    end
end
