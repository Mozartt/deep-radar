clear, clc, close all

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

%% area of Interset
X1 = (-10:0.05:10) + P_trgt(1);
Y1 = (-10:0.05:10) + P_trgt(2);
Z1 = P_trgt(3); % linspace(-10, 10, 20) + P_trgt(3);
% matched Filter for each point in area of interest
RESULT = NaN(length(X1),length(Y1));
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
            g = abs( sum( sum( conj(XX) .* y_ell ) ) );
            RESULT(jy,jx,jz) =  g;
        end
    end
end
close(h)
toc
%%% PLOTS %%%%%%%%%%%
figure
imagesc(X1,Y1,20*log10(RESULT))
hold on
plot(P_trgt(1),P_trgt(2),'ko',MarkerSize=10)
axis('equal')
xlabel('X [meters]')
ylabel('Y [meters]')
title('Intensity in [dB]')
colorbar

figure
contour(X1,Y1,20*log10(RESULT))
hold on
plot(P_trgt(1),P_trgt(2),'ko',MarkerSize=10)
axis('equal')
xlabel('X [meters]')
ylabel('Y [meters]')
title('Intensity in [dB]')
colorbar

figure
[~,Ix] = min(abs(X1 - P_trgt(1)));
plot(Y1,20*log10(RESULT(:,Ix)))
xline(P_trgt(2),'k-','True Y location')
grid, grid minor
title('Intersection along true Trgt X')
xlabel('Y [meters]')
ylabel('Intensity [dB]')

figure
[~,Iy] = min(abs(Y1 - P_trgt(2)));
plot(X1,20*log10(RESULT(Iy,:)))
xline(P_trgt(1),'-k','True X location')
grid, grid minor
title('Intersection along true Trgt Y')
xlabel('X [meters]')
ylabel('Intensity [dB]')

