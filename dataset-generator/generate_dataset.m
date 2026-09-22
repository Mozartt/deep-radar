clear;
clc;
close all;

addpath("../simulator/")

%% =========================================================
% Dataset generation for radar localization
%
% INPUT:
%   alpha    : target amplitude
%   SNR      : SNR in dB
%
% OUTPUT:
%   
%
%% =========================================================
%% -------------------------------
% Transmission parameters
%% -------------------------------
tran_config.c = 3e8; % light speed [m/S]
tran_config.fc = 2e9; % center freq [Hz]
tran_config.BW = 0.2e9; % Band width [Hz]
tran_config.M = 40; % Number of receivers
tran_config.Tc = 20e-6; % Chip length [sec]
tran_config.a = tran_config.BW / tran_config.Tc;
tran_config.Fs = 50e6; % sampling freq [Hz]
tran_config.Ts = 1 / tran_config.Fs; % sampling period
tran_config.N = round(tran_config.Tc * tran_config.Fs); % number of samples
tran_config.n = 0 : tran_config.N-1;
tran_config.recievers_circle_radius = 100; % receivers are ordered in a circle
tran_config.p_trnsmt = zeros(3,1); % Transmitter location [x;y;z] [meters]

theta = 2 * pi * (0 : tran_config.M-1)./tran_config.M; % radians
R = tran_config.recievers_circle_radius;
tran_config.q = R*[cos(theta); sin(theta); zeros(size(theta)) ]; % antenna locations [meters]

%% -------------------------------
% Dataset parameters
%% -------------------------------

numSamples = 80000;
numOfTargets = 4; % number of targets per sample

% Signal parameters
alphaRange = [1, 1];
snrRange   = [-5, 20]; %dB

% Output folder
datasetDir = "D:\radar-dataset-multi-targets-2\";

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
    [targets, K] = sample_targets_simple(0, radius, zRange(1), zRange(2), numOfTargets);
    
    %% ---------------------------------
    % Random radar conditions
    %% ---------------------------------

    alpha = rand_uniform(alphaRange);
    SNR = snrRange(1) + (snrRange(2)-snrRange(1)) * rand();

    %% ---------------------------------
    % Generate heatmap
    %% ---------------------------------

    [y_clean, y_ell, tau, phi] = get_radar_response_noisy(targets, alpha, SNR, size(targets,1), K, tran_config);

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
        'numTargets', single(K), ...
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