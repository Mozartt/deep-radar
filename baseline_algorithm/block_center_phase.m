function phi = block_center_phase( ...
    P_hat, m_idx, n_idx, tran_config)

    c  = tran_config.c;
    fc = tran_config.fc;
    a  = tran_config.a;
    Ts = tran_config.Ts;

    q  = tran_config.q;
    tx = tran_config.p_trnsmt;

    % Center receiver of this receiver block
    mc = m_idx(round((length(m_idx)+1)/2));

    % Center fast-time sample
    nc = mean(n_idx);

    tc = (nc-1)*Ts;

    % Delay corresponding to center receiver
    tau_c = norm(P_hat-tx)/c + ...
            norm(q(:,mc)-P_hat)/c;

    % Phase of conjugated MF kernel
    phi = ...
          2*pi*fc*tau_c ...
        - pi*a*tau_c^2 ...
        + 2*pi*a*tau_c*tc;

end