clear, clc, close all
addpath("simulator\")
%% This code calculates the MF cost over a single block out of the 
%% Time-Receiver plane, Then it apllies phase centering term to the matched filter cost
%% And plots the resulting intensity maps to see if the fast osilating terms reduced.


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

P_trnsmt = zeros(3,1); % Transmitter location [x;y;z] [meters]
P_trgt = [200; 300; 500]; % Target location

% tau = c*norm(P_trgt - P_trnsmt) + c*vecnorm(q - P_trgt); % propagation time
alfa = 1; % attenuation
SNR=30;
K=1;
[y_ell, tau, phi] = get_radar_response(P_trgt,alfa, tran_config);

%% m,n sub domain 
m_idx = 1:10;
n_idx = 1:125;
y_ell = y_ell(m_idx, n_idx);

X1 = (-10:0.05:10) + P_trgt(1);
Y1 = (-10:0.05:10) + P_trgt(2);
Z1 = P_trgt(3); % linspace(-10, 10, 20) + P_trgt(3);
% matched Filter for each point in area of interest
RESULT = NaN(length(Y1),length(X1));
phase_centering = zeros(length(Y1), length(X1), length(Z1));
h = waitbar(0, 'Please wait...');
jx = 0;
tic
for Px = X1
    jx = jx + 1;
    waitbar(jx/length(X1),h)
    jy = 0;
    for Py = Y1
        jy = jy + 1;
        jz = 0;
        for Pz = Z1
            jz = jz + 1;
            P_hat = [Px;Py;Pz]; % Target location
            % tau_hat = c*norm(P_hat - P_trnsmt) + c*vecnorm(q - P_hat); % propagation time
            % beta_hat = exp(-1i * 2 * pi * fc * tau_hat).*exp( 1i * pi * a * tau_hat.^2);
            % XX = diag(beta_hat) * exp( - 1i * 2 * pi * a * tau_hat' * Ts * n);
            [XX, tau, phi] = get_radar_response(P_hat, alfa, tran_config);

            XX = XX(m_idx, n_idx);
            g = sum(sum(conj(XX) .* y_ell));
            RESULT(jy,jx,jz) =  g;
            phase_centering(jy,jx,jz) = block_center_phase(P_hat, m_idx, n_idx, tran_config);
        end
    end
end
close(h)
toc

RESULT_centered = RESULT .* exp(-1i * phase_centering);
figure;

subplot(2,2,1)
imagesc(X1,Y1,real(RESULT))
axis xy equal
title('Real\{S_b\}')
colorbar

subplot(2,2,2)
imagesc(X1,Y1,real(RESULT_centered))
axis xy equal
title('Real\{\tilde S_b\}')
colorbar

subplot(2,2,3)
imagesc(X1,Y1,angle(RESULT))
axis xy equal
title('Phase S_b')
colorbar

subplot(2,2,4)
imagesc(X1,Y1,angle(RESULT_centered))
axis xy equal
title('Phase \tilde S_b')
colorbar

figure;
[~,iy] = min(abs(Y1-P_trgt(2)));
subplot(2,1,1)
plot(X1, real(RESULT(iy,:)), ...
    'DisplayName','Original');
hold on

plot(X1, real(RESULT_centered(iy,:)), ...
    'DisplayName','Phase centered');
grid on
xlabel('X [m]')
ylabel('Real part')
legend
title('Complex image variation along Y = Y_{target}')


subplot(2,1,2)

phase_original = unwrap(angle(RESULT(iy,:)));
phase_centered = unwrap(angle(RESULT_centered(iy,:)));

plot(X1, phase_original, ...
    'DisplayName','Original');
hold on

plot(X1, phase_centered, ...
    'DisplayName','Phase centered');

grid on
xlabel('X [m]')
ylabel('Unwrapped phase [rad]')
legend
title('Unwrapped phase')