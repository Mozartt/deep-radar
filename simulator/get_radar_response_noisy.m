function [y_clean, y_ell, tau_all, phi_all] = get_radar_response_noisy(targets, alfa, SNR_dB, maxTargets, K, tran_config)

% common
c = tran_config.c; % light speed [m/S]
fc = tran_config.fc;
M = tran_config.M; % Number of receivers
Tc = tran_config.Tc; % Chip length [sec]
a = tran_config.a;
Ts = tran_config.Ts; % sampling period
N = tran_config.N; % number of samples
n = tran_config.n;


P_trnsmt = tran_config.p_trnsmt; % Transmitter location [x;y;z] [meters]
q = tran_config.q; % antenna locations [meters]

[y_total, tau_all, phi_all] = deal(0, zeros(M,maxTargets), zeros(M,maxTargets));

for k = 1:K
    P_trgt = targets(k, :).'; % Target location [x;y;z] [meters]
    [y_ell, tau, phi] = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c,Tc);
    y_ell = alfa * y_ell;
    y_total = y_total + y_ell;
    tau_all(:, k) = tau;
    phi_all(:, k) = phi;
end

y_clean = y_total;

% Signal power
%signal_power = mean(abs(y_ell(:)).^2);
signal_power = 1;

% Noise power
noise_power = N * signal_power / (10^(SNR_dB/10));

% Complex Gaussian noise
noise = sqrt(noise_power/2) * ...
    (randn(size(y_ell)) + 1i*randn(size(y_ell)));

% Noisy received signal
y_ell = y_ell + noise;

    function [y_ell, tau, phi] = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c,Tc)
        tau = norm(P_trgt - P_trnsmt)/c + vecnorm(q - P_trgt)./c; % propagation time
        beta = exp( - 1i * 2 * pi * fc * tau).*exp( 1i * pi * a * tau.^2);
        phi = angle(beta);
        y_ell = diag(beta) * exp( - 1i * 2 * pi * a * tau' * Ts * n); % observed data
        t = Ts * n;
        win = (t >= tau.') & (t <= Tc);
        y_ell = y_ell .* win;
        y_ell = y_ell(:,N-999:end);
    end
end


