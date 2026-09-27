function [y_ell, tau, phi] = get_radar_response(P_trgt, alfa, tran_config)

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

[y_ell, tau, phi] = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c);
y_ell = alfa * y_ell;

function [y_ell, tau, phi] = Radar_Response(P_trnsmt,q,P_trgt,fc,a,Ts,n,c)
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


