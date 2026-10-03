clear, clc, close all
addpath("Simulator/")
%% -------------------------------
% Transmission parameters
%% -------------------------------
tran_config.c = 3e8; % light speed [m/S]
tran_config.fc = 2e9; % center freq [Hz]
tran_config.BW = 0.2e9; % Band width [Hz]
tran_config.M = 40; % Number of receivers
tran_config.Tc = 20e-6; % Chip length [sec]
tran_config.a = tran_config.BW / tran_config.Tc;
tran_config.alfa = 1;
tran_config.Fs = 50e6; % sampling freq [Hz]
tran_config.Ts = 1 / tran_config.Fs; % sampling period
tran_config.N = round(tran_config.Tc * tran_config.Fs); % number of samples
tran_config.n = 0 : tran_config.N-1;
tran_config.recievers_circle_radius = 100; % receivers are ordered in a circle
tran_config.p_trnsmt = zeros(3,1); % Transmitter location [x;y;z] [meters]
theta = 2 * pi * (0 : tran_config.M-1)./tran_config.M; % radians
R = tran_config.recievers_circle_radius;
tran_config.q = R*[cos(theta); sin(theta); zeros(size(theta)) ]; % antenna locations [meters]
tran_config.p_trnsmt = zeros(3,1); % Transmitter location [x;y;z] [meters]

% define the ground truth position
p_trgt = [200; 300; 500]; % Target location

[signal, tau, phi] = get_radar_response(p_trgt, tran_config);

% define the search domain
z0 = 500;          % example fixed-z plane
x_lim = [-150 150];
y_lim = [-150 150];
dx = 0.05;

% define algorithm configuration
opts.interp_method = 'linear';
opts.verbose = true;

tic
% run algorithm
[mf_boag, x, y, info] = fast_matched_filter( ...
    signal, ...
    tran_config, ...
    x_lim, ...
    y_lim, ...
    z0, ...
    dx, ...
    'interp_method','linear', ...
    'verbose',true);
elapsed = toc;

figure;
imagesc(x, y, abs(mf_boag));
axis image;
set(gca,'YDir','normal');
colorbar;

xlabel('x [m]');
ylabel('y [m]');
title('|MF| - Boag bottom-up');