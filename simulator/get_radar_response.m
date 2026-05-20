function [y_ell] = get_radar_response(P_trgt, alfa, SNR_dB)

c = 3e8; % light speed [m/S]
fc = 2e9; % center freq [Hz]
BW = 0.2e9; % Band width [Hz]
Tc = 0.2e-6; % Chip length [sec]
a = BW / Tc;
M = 40; % Number of receivers
Fs = 2 * BW; % sampling freq [Hz]
Ts = 1 / Fs; % sampling period
N = round(Tc * Fs); % number of samples
n = 0 : N-1;

P_trnsmt = zeros(3,1); % Transmitter location [x;y;z] [meters]
theta = 2 * pi * (0 : M-1)./M; % radians
q = 100*[cos(theta); sin(theta); zeros(size(theta)) ]; % antenna locations [meters

% tau = c*norm(P_trgt - P_trnsmt) + c*vecnorm(q - P_trgt); % propagation time
% beta = exp( - 1i * 2 * pi * fc * tau).*exp( 1i * pi * a * tau.^2);
% y_ell = alfa * diag(beta) * exp( - 1i * 2 * pi * a * tau' * Ts * n); % observed data
y_ell = alfa * Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c);

% Signal power
%signal_power = mean(abs(y_ell(:)).^2);

% Noise power
%noise_power = signal_power / (10^(SNR_dB/10));

% Complex Gaussian noise
%noise = sqrt(noise_power/2) * ...
%    (randn(size(y_ell)) + 1i*randn(size(y_ell)));

% Noisy received signal
%y_ell = y_ell + noise;

function y_ell = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c)
    tau = norm(P_trgt - P_trnsmt)/c + vecnorm(q - P_trgt)./c; % propagation time
    beta = exp( - 1i * 2 * pi * fc * tau).*exp( 1i * pi * a * tau.^2);
    y_ell = diag(beta) * exp( - 1i * 2 * pi * a * tau' * Ts * n); % observed data
end
end


