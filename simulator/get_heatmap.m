function [y_ell_clean,heatmap] = get_heatmap(P_trgt, alfa, SNR_dB)

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
y_ell_clean = y_ell; % clean signal without noise

% Signal power
signal_power = mean(abs(y_ell(:)).^2);

% Noise power
noise_power = signal_power / (10^(SNR_dB/10));

% Complex Gaussian noise
noise = sqrt(noise_power/2) * ...
    (randn(size(y_ell)) + 1i*randn(size(y_ell)));

% Noisy received signal
y_ell = y_ell + noise;

%% area of Interset
X1 = (-10:0.05:10) + P_trgt(1);
Y1 = (-10:0.05:10) + P_trgt(2);
Z1 = P_trgt(3); % linspace(-10, 10, 20) + P_trgt(3);
% matched Filter for each point in area of interest
RESULT = NaN(length(X1),length(Y1));
% h = waitbar(0, 'Please wait...');
jx = 0;
tic
for Px = X1
    jx = jx + 1;
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
            XX = Radar_Response(P_trnsmt,q,P_hat,fc,a,Ts,n,c);
            g = abs( sum( sum( conj(XX) .* y_ell ) ) );
            RESULT(jy,jx,jz) =  g;
        end
    end
end
% close(h)
toc
heatmap = 20*log10(RESULT + 1e-12);
%%% PLOTS %%%%%%%%%%%
% figure
% imagesc(X1,Y1,20*log10(RESULT))
% hold on
% plot(P_trgt(1),P_trgt(2),'ko',MarkerSize=10)
% axis('equal')
% xlabel('X [meters]')
% ylabel('Y [meters]')
% title('Intensity in [dB]')
% colorbar

% figure
% contour(X1,Y1,20*log10(RESULT))
% hold on
% plot(P_trgt(1),P_trgt(2),'ko',MarkerSize=10)
% axis('equal')
% xlabel('X [meters]')
% ylabel('Y [meters]')
% title('Intensity in [dB]')
% colorbar

% figure
% [~,Ix] = min(abs(X1 - P_trgt(1)));
% plot(Y1,20*log10(RESULT(:,Ix)))
% xline(P_trgt(2),'k-','True Y location')
% grid, grid minor
% title('Intersection along true Trgt X')
% xlabel('Y [meters]')
% ylabel('Intensity [dB]')
% 
% figure
% [~,Iy] = min(abs(Y1 - P_trgt(2)));
% plot(X1,20*log10(RESULT(Iy,:)))
% xline(P_trgt(1),'-k','True X location')
% grid, grid minor
% title('Intersection along true Trgt Y')
% xlabel('X [meters]')
% ylabel('Intensity [dB]')


function y_ell = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c)
    tau = norm(P_trgt - P_trnsmt)/c + vecnorm(q - P_trgt)./c; % propagation time
    beta = exp( - 1i * 2 * pi * fc * tau).*exp( 1i * pi * a * tau.^2);
    y_ell = diag(beta) * exp( - 1i * 2 * pi * a * tau' * Ts * n); % observed data
end
end


