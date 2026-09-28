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

Xfine = (-10:0.05:10) + P_trgt(1);
Yfine = (-10:0.05:10) + P_trgt(2);
Zfine = P_trgt(3); % linspace(-10, 10, 20) + P_trgt(3);

% sub-domains
m_idx = 1:10;
n_idx = 1:125;

S_exact = NaN(length(Yfine),length(Xfine));
phase_centering = zeros(length(Yfine), length(Xfine), length(Zfine));
h = waitbar(0, 'Please wait...');
jx = 0;
tic
for Px = Xfine
    jx = jx + 1;
    waitbar(jx/length(Xfine),h)
    jy = 0;
    for Py = Yfine
        jy = jy + 1;
        jz = 0;
        for Pz = Zfine
            jz = jz + 1;
            P_hat = [Px;Py;Pz]; % Target location
            % tau_hat = c*norm(P_hat - P_trnsmt) + c*vecnorm(q - P_hat); % propagation time
            % beta_hat = exp(-1i * 2 * pi * fc * tau_hat).*exp( 1i * pi * a * tau_hat.^2);
            % XX = diag(beta_hat) * exp( - 1i * 2 * pi * a * tau_hat' * Ts * n);
            [XX, tau, phi] = get_radar_response(P_hat, alfa, tran_config);

            XX = XX(m_idx, n_idx);
            g = sum(sum(conj(XX) .* y_ell));
            S_exact(jy,jx,jz) =  g;
            phase_centering(jy,jx,jz) = block_center_phase(P_hat, m_idx, n_idx, tran_config);
        end
    end
end
close(h)
toc

skip = 10;       % 10 * 0.05 = 0.5 m

ix_c = 1:skip:length(Xfine);
iy_c = 1:skip:length(Yfine);

Xcoarse = Xfine(ix_c);
Ycoarse = Yfine(iy_c);

S_coarse = S_exact(iy_c,ix_c,:);
phase_centering_coarse = phase_centering(iy_c, ix_c,:);

S_centered_coarse = ...
    S_coarse .* exp(-1i*phase_centering_coarse);

% Interpolate
[Xc,Yc] = meshgrid(Xcoarse,Ycoarse);
[Xf,Yf] = meshgrid(Xfine,Yfine);

Sr = interp2( ...
    Xc,Yc,real(S_centered_coarse), ...
    Xf,Yf,'spline');

Si = interp2( ...
    Xc,Yc,imag(S_centered_coarse), ...
    Xf,Yf,'spline');

S_centered_interp = Sr + 1i*Si;

S_boag = ...
    S_centered_interp .* exp(1i*phase_centering);

err_boag = ...
    norm(S_exact(:)-S_boag(:)) / ...
    norm(S_exact(:));

fprintf('Boag relative error = %.4e\n',err_boag);

Sr = interp2( ...
    Xc,Yc,real(S_coarse), ...
    Xf,Yf,'spline');

Si = interp2( ...
    Xc,Yc,imag(S_coarse), ...
    Xf,Yf,'spline');

S_naive = Sr + 1i*Si;

err_naive = ...
    norm(S_exact(:)-S_naive(:)) / ...
    norm(S_exact(:));

fprintf('Naive relative error = %.4e\n',err_naive);

figure;
imagesc(Xfine,Yfine,abs(S_boag));
axis xy;
xlabel('X [m]');
ylabel('Y [m]');
title('Boag Interpolated Response Magnitude');

figure;
imagesc(Xfine,Yfine,abs(S_exact));
axis xy;
xlabel('X [m]');
ylabel('Y [m]');
title('Exact Response Magnitude');
colorbar;