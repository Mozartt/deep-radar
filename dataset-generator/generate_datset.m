clear;
clc;
close all;

addpath("../simulator/")

%% =========================================================
% Dataset generation for radar heatmap learning
%
% Assumes you already implemented:
%
%   heatmap = get_heatmap(p_target, alpha, SNR)
%
% INPUT:
%   p_target : [x;y;z]
%   alpha    : target amplitude
%   SNR      : SNR in dB
%
% OUTPUT:
%   heatmap  : 2D matrix
%
%% =========================================================

%% -------------------------------
% Dataset parameters
%% -------------------------------

numSamples = 1;

% Spatial region for target generation
xRange = [100, 1000];
yRange = [100, 1000];
zRange = [500, 500];

% Signal parameters
alphaRange = [0.5, 2.0];
snrRange   = [5, 40];

% Output folder
datasetDir = "D:\radar-dataset3\";

if ~exist(datasetDir, 'dir')
    mkdir(datasetDir);
end

%% -------------------------------
% Preallocate labels
%% -------------------------------

targetXYZ = zeros(numSamples, 3);
alphaVec  = zeros(numSamples, 1);
snrVec    = zeros(numSamples, 1);

%% -------------------------------
% Generate dataset
%% -------------------------------

fprintf('Generating dataset...\n');

parfor i = 1:numSamples

    %% ---------------------------------
    % Random target location
    %% ---------------------------------

    x = rand_uniform(xRange);
    y = rand_uniform(yRange);
    z = rand_uniform(zRange);

    p_target = [x; y; z];
    
    %% ---------------------------------
    % Random radar conditions
    %% ---------------------------------

    alpha = rand_uniform(alphaRange);
    SNR   = rand_uniform(snrRange);

    %% ---------------------------------
    % Generate heatmap
    %% ---------------------------------

    [y_ell_clean, heatmap] = get_heatmap(p_target, alpha, SNR);

    %% ---------------------------------
    % Normalize heatmap
    %% ---------------------------------

    %heatmap = normalize_heatmap(heatmap);

    %% ---------------------------------
    % Save sample
    %% ---------------------------------

    sample = struct( ...
        'heatmap', single(heatmap), ...
        'y_ell_clean', single(y_ell_clean), ...
        'target_xyz', single([x y z]), ...
        'alpha', single(alpha), ...
        'SNR', single(SNR), ...
        'sample_id', i ...
    );

    parsave_sample( ...
        fullfile(datasetDir, sprintf('sample_%06d.mat', i)), ...
        sample ...
    );

    %% ---------------------------------
    % Save labels also globally
    %% ---------------------------------

    targetXYZ(i,:) = [x y z];
    alphaVec(i) = alpha;
    snrVec(i) = SNR;

end

%% -------------------------------
% Save dataset metadata
%% -------------------------------

metadata.numSamples = numSamples;

metadata.xRange = xRange;
metadata.yRange = yRange;
metadata.zRange = zRange;

metadata.alphaRange = alphaRange;
metadata.snrRange = snrRange;

metadata.targetXYZ = targetXYZ;

metadata.alphaVec = alphaVec;
metadata.snrVec = snrVec;

save( ...
    fullfile(datasetDir, 'dataset_metadata.mat'), ...
    'metadata', ...
    '-v7.3' ...
);

fprintf('\nDataset generation completed.\n');

%% =========================================================
%% Helper functions
%% =========================================================

function val = rand_uniform(range)

    val = range(1) + rand() * (range(2)-range(1));

end

function hm = normalize_heatmap(hm)

    hm = hm - min(hm(:));

    hm = hm ./ (max(hm(:)) + 1e-12);

end

function parsave_sample(filePath, sample)

    save(filePath, 'sample', '-v7.3');

end