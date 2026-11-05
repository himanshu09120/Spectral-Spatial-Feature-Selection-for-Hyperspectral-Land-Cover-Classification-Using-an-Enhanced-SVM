function Main(mode)
%MAIN Portable MATLAB launcher. Main('validate-data') performs no training.
% Set FRSS_PYTHON to an absolute interpreter path if not using project .venv.
% Main explicitly starts the full experiment; other modes never train.
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
    if strcmp(mode, 'run')
        command = [command ' --allow-training'];
    end
    [status, output] = system(command, '-echo');
    if status ~= 0
        error('FRSS:PythonFailed', 'Python exited with status %d.\n%s', status, output);
    end
    if strcmp(mode, 'run')
        settings = jsondecode(fileread(fullfile(root, 'config', 'config.json')));
        resultRoot = fullfile(root, settings.output_dir);
        disp(readtable(fullfile(resultRoot, 'metrics', 'summary.csv')));
        disp(readtable(fullfile(resultRoot, 'metrics', 'per_class_summary.csv')));
        for index = 1:numel(settings.protocols)
            protocol = settings.protocols{index};
            images = {'accuracy_comparison.png', 'macro_f1_comparison.png', ...
                      'runtime_vs_features.png'};
            for imageIndex = 1:numel(images)
                imagePath = fullfile(resultRoot, 'figures', protocol, images{imageIndex});
                figure('Name', [protocol ' - ' images{imageIndex}], 'NumberTitle', 'off');
                image(imread(imagePath)); axis image off;
            end
            if ~isempty(settings.execution.map_seeds)
                mapSeed = settings.execution.map_seeds(1);
                detailImages = {'E1_confusion_normalized.png', 'E5_confusion_normalized.png', 'E5_comparison.png'};
                for imageIndex = 1:numel(detailImages)
                    imagePath = fullfile(resultRoot, 'figures', protocol, ...
                        sprintf('seed_%d', mapSeed), detailImages{imageIndex});
                    if isfile(imagePath)
                        figure('Name', detailImages{imageIndex}, 'NumberTitle', 'off');
                        image(imread(imagePath)); axis image off;
                    end
                end
            end
        end
        fprintf('All metrics, confusion matrices and maps: %s\n', resultRoot);
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
