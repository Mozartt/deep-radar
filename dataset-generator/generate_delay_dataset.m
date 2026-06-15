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

numSamples = 10000;

% Spatial region for target generation

zRange = [300, 300];

% Signal parameters
alphaRange = [1, 1];
snrRange   = [-5, 20]; %dB

% Output folder
datasetDir = "D:\radar-dataset-noisy\";

testDir  = fullfile(datasetDir, 'test');
valDir   = fullfile(datasetDir, 'validation');
trainDir = fullfile(datasetDir, 'train');

for d = {testDir, valDir, trainDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

% Split boundaries (first 15% test, next 15% validation, rest train)
nTest  = round(0.15 * numSamples);
nVal   = round(0.15 * numSamples);
% nTrain = numSamples - nTest - nVal  (remainder)

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
radius = 150;
theta = 2 * pi * rand(numSamples, 1);
r = radius * sqrt(rand(numSamples, 1));

for i = 1:numSamples

    %% ---------------------------------
    % Random target location
    %% ---------------------------------

    x = r(i) .* cos(theta(i));
    y = r(i) .* sin(theta(i));
    z = rand_uniform(zRange);
    p_target = [x; y; z];
    
    %% ---------------------------------
    % Random radar conditions
    %% ---------------------------------

    alpha = rand_uniform(alphaRange);
    SNR = snrRange(1) + (snrRange(2)-snrRange(1)) * rand();

    %% ---------------------------------
    % Generate heatmap
    %% ---------------------------------

    [y_ell, tau, phi] = get_radar_response_noisy(p_target, alpha, SNR);

    %% ---------------------------------
    % Normalize heatmap
    %% ---------------------------------

    heatmap = [];

    %% ---------------------------------
    % Save sample
    %% ---------------------------------

    sample = struct( ...
        'y_ell', single(y_ell), ...
        'heatmap', single(heatmap), ...
        'tau', single(tau), ...
        'phi', single(phi), ...
        'target_xyz', single([x y z]), ...
        'alpha', single(alpha), ...
        'SNR', single(SNR), ...
        'sample_id', i ...
    );

    %% ---------------------------------
    % Determine split folder
    %% ---------------------------------

    if i <= nTest
        splitDir = testDir;
    elseif i <= nTest + nVal
        splitDir = valDir;
    else
        splitDir = trainDir;
    end

    parsave_sample( ...
        fullfile(splitDir, sprintf('sample_%06d.mat', i)), ...
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

%metadata.xRange = xRange;
%metadata.yRange = yRange;
%metadata.zRange = zRange;

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

function parsave_sample(filePath, sample)

    save(filePath, 'sample', '-v7.3');

end