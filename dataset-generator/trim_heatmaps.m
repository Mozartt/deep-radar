clear;
clc;

%% =========================================================
% Trim heatmaps in a dataset to a centred square crop.
%
% For every sample_*.mat in inputDir the script:
%   1. Loads sample.heatmap
%   2. Finds the centre row/col of the 2-D heatmap
%   3. Keeps a (SIZE x SIZE) square around that centre
%   4. Stores the modified sample to outputDir under the
%      same filename, leaving all other fields unchanged.
%
% Adjust inputDir, outputDir and SIZE below before running.
%% =========================================================

%% ----------------------------
% Parameters  <-- edit here
%% ----------------------------

inputDir  = "D:\radar-dataset2\test";
outputDir = "D:\radar-dataset3\test";
SIZE      = 64;      % side length of the square crop (pixels)

%% ----------------------------
% Setup
%% ----------------------------

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

files = dir(fullfile(inputDir, 'sample_*.mat'));
numFiles = numel(files);

if numFiles == 0
    error('No sample_*.mat files found in: %s', inputDir);
end

fprintf('Found %d samples. Trimming heatmaps to %dx%d ...\n', ...
    numFiles, SIZE, SIZE);

%% ----------------------------
% Process
%% ----------------------------

half = floor(SIZE / 2);

parfor k = 1 : numFiles

    srcPath = fullfile(inputDir,  files(k).name);
    dstPath = fullfile(outputDir, files(k).name);

    %% Load
    data   = load(srcPath, 'sample');
    sample = data.sample;

    %% Crop heatmap
    hm = sample.heatmap;          % [H x W], single

    [H, W] = size(hm);

    cRow = round(H / 2);
    cCol = round(W / 2);

    r1 = max(cRow - half, 1);
    r2 = min(cRow + half, H);
    c1 = max(cCol - half, 1);
    c2 = min(cCol + half, W);

    sample.heatmap = hm(r1:r2, c1:c2);

    %% Save (parfor-safe)
    parsave_trimmed(dstPath, sample);

end

fprintf('Done. Trimmed samples saved to: %s\n', outputDir);

%% =========================================================
%% Helper
%% =========================================================

function parsave_trimmed(filePath, sample)
    save(filePath, 'sample', '-v7.3');
end
