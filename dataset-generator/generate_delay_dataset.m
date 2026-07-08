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

numSamples = 80000;
numOfTargets = 4; % number of targets per sample

% Signal parameters
alphaRange = [1, 1];
snrRange   = [-5, 20]; %dB

% Output folder
datasetDir = "D:\radar-dataset-multi-targets\";

testDir  = fullfile(datasetDir, 'test');
valDir   = fullfile(datasetDir, 'validation');
trainDir = fullfile(datasetDir, 'train');

for d = {testDir, valDir, trainDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

% Split boundaries (15% test, 15% validation, rest train) — randomly shuffled
nTest  = round(0.15 * numSamples);
nVal   = round(0.15 * numSamples);
% nTrain = numSamples - nTest - nVal  (remainder)

perm = randperm(numSamples);          % random shuffle of all sample indices
splitLabel = zeros(numSamples, 1);    % 1=test, 2=val, 3=train
splitLabel(perm(1:nTest))              = 1;
splitLabel(perm(nTest+1:nTest+nVal))   = 2;
splitLabel(perm(nTest+nVal+1:end))     = 3;

%% -------------------------------
% Preallocate labels
%% -------------------------------

targets_all = cell(numSamples, 1);
targets_num = cell(numSamples,1);
alphaVec  = zeros(numSamples, 1);
snrVec    = zeros(numSamples, 1);

%% -------------------------------
% Generate dataset
%% -------------------------------

fprintf('Generating dataset...\n');
radius = 150;
zRange = [200 300];

for i = 1:numSamples

    %% ---------------------------------
    % Random target location
    %% ---------------------------------
    [targets, quadrants] = sample_targets_by_quadrants(0, radius, zRange(1), zRange(2), numOfTargets);
    
    %% ---------------------------------
    % Random radar conditions
    %% ---------------------------------

    alpha = rand_uniform(alphaRange);
    SNR = snrRange(1) + (snrRange(2)-snrRange(1)) * rand();

    %% ---------------------------------
    % Generate heatmap
    %% ---------------------------------

    [y_clean, y_ell, tau, phi] = get_radar_response_noisy(targets, alpha, SNR, size(targets,1));

    %% ---------------------------------
    % Normalize heatmap
    %% ---------------------------------

    heatmap = [];

    %% ---------------------------------
    % Save sample
    %% ---------------------------------

    sample = struct( ...
        'y_clean', single(y_clean), ...
        'y_ell', single(y_ell), ...
        'heatmap', single(heatmap), ...
        'tau', single(tau), ...
        'phi', single(phi), ...
        'target_xyz', single(targets), ...
        'alpha', single(alpha), ...
        'SNR', single(SNR), ...
        'sample_id', i ...
    );

    %% ---------------------------------
    % Determine split folder
    %% ---------------------------------

    switch splitLabel(i)
        case 1, splitDir = testDir;
        case 2, splitDir = valDir;
        case 3, splitDir = trainDir;
    end

    parsave_sample( ...
        fullfile(splitDir, sprintf('sample_%06d.mat', i)), ...
        sample ...
    );

    %% ---------------------------------
    % Save labels also globally
    %% ---------------------------------

    targets_all{i} = targets;
    targets_num{i} = size(targets,1);
    alphaVec(i) = alpha;
    snrVec(i) = SNR;

end

%% -------------------------------
% Save dataset metadata
%% -------------------------------

metadata.numSamples = numSamples;

metadata.alphaRange = alphaRange;
metadata.snrRange = snrRange;

metadata.targets = targets_all;
metadata.targets_num = targets_num;
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

function parsave_sample(filePath, sample)

    save(filePath, 'sample', '-v7.3');

end