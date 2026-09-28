function [mf_img, x_vec, y_vec, info] = fast_matched_filter(signal,tran_config, x_lim, y_lim, z0, dx, opts)
%
% Full bottom-up multilevel matched-filter imaging.
%
% signal : [M x N] complex deramped FMCW data
% rx_pos : [M x 3] receiver positions
% tx_pos : [1 x 3] transmitter position
%
% tran_config fields:
%   .c
%   .fc
%   .a
%   .Ts
%
% x_lim = [xmin xmax]
% y_lim = [ymin ymax]
% z0    = fixed imaging height
% dx    = final spatial grid spacing
%
% opts.interp_method     = 'linear' / 'spline'
% opts.verbose           = true / false
% opts.min_leaf_pixels   = optional minimum coarse image size
%
% Output:
%   mf_img : complex MF image [Ny x Nx]
%   x_vec, y_vec
%
% The image magnitude is abs(mf_img).

    arguments
        signal
        tran_config
        x_lim
        y_lim
        z0
        dx
        opts.interp_method char = 'linear'
        opts.verbose logical = true
        opts.min_leaf_pixels double = 1
    end

    signal = double(signal);

    [M, N] = size(signal);

    rx_pos = tran_config.q';
    tx_pos = tran_config.p_trnsmt';

    %% ------------------------------------------------------------
    % Final spatial grid
    %% ------------------------------------------------------------

    x_vec = x_lim(1):dx:x_lim(2);
    y_vec = y_lim(1):dx:y_lim(2);

    Nx_final = length(x_vec);
    Ny_final = length(y_vec);

    total_samples = M * N;

    if opts.verbose
        fprintf('Boag bottom-up imaging\n');
        fprintf('Data: %d receivers x %d samples = %d samples\n', ...
            M, N, total_samples);
        fprintf('Final image: %d x %d = %d pixels\n', ...
            Ny_final, Nx_final, Nx_final * Ny_final);
    end

    %% ------------------------------------------------------------
    % Level 0
    %
    % One node for every radar datum.
    %
    % A leaf centered on exactly (m,n) has
    %
    %   I_tilde = y(m,n)
    %
    % since phi(m,n)-phi_center = 0.
    %% ------------------------------------------------------------

    nodes = cell(M, N);

    leaf_nx = max(opts.min_leaf_pixels, ...
        ceil(Nx_final / sqrt(total_samples)));

    leaf_ny = max(opts.min_leaf_pixels, ...
        ceil(Ny_final / sqrt(total_samples)));

    leaf_nx = min(leaf_nx, Nx_final);
    leaf_ny = min(leaf_ny, Ny_final);

    for m = 1:M
        for n = 1:N

            node.m_lo = m;
            node.m_hi = m;

            node.n_lo = n;
            node.n_hi = n;

            node.m_center = m;
            node.t_center = (n-1) * tran_config.Ts;

            node.n_samples = 1;

            % Phase-centered leaf image is constant.
            node.img = signal(m,n) * ...
                ones(leaf_ny, leaf_nx);

            nodes{m,n} = node;
        end
    end

    %% ------------------------------------------------------------
    % Bottom-up hierarchy
    %% ------------------------------------------------------------

    level = 0;

    level_stats = [];

    while numel(nodes) > 1

        level = level + 1;

        [Nm_old, Nn_old] = size(nodes);

        Nm_new = ceil(Nm_old / 2);
        Nn_new = ceil(Nn_old / 2);

        if opts.verbose
            fprintf('\nLevel %d\n', level);
            fprintf('  %d x %d blocks -> %d x %d blocks\n', ...
                Nm_old, Nn_old, Nm_new, Nn_new);
        end

        new_nodes = cell(Nm_new, Nn_new);

        tic;

        for pm = 1:Nm_new
            for pn = 1:Nn_new

                %% ------------------------------------------------
                % Collect up to four children
                %% ------------------------------------------------

                children = {};

                m_inds = [2*pm-1, 2*pm];
                n_inds = [2*pn-1, 2*pn];

                for ii = 1:2
                    for jj = 1:2

                        mi = m_inds(ii);
                        ni = n_inds(jj);

                        if mi <= Nm_old && ni <= Nn_old
                            if ~isempty(nodes{mi,ni})
                                children{end+1} = nodes{mi,ni}; %#ok<AGROW>
                            end
                        end

                    end
                end

                if isempty(children)
                    continue;
                end

                %% ------------------------------------------------
                % Parent data domain
                %% ------------------------------------------------

                m_lo = inf;
                m_hi = -inf;

                n_lo = inf;
                n_hi = -inf;

                nsamp = 0;

                for q = 1:length(children)
                    m_lo = min(m_lo, children{q}.m_lo);
                    m_hi = max(m_hi, children{q}.m_hi);

                    n_lo = min(n_lo, children{q}.n_lo);
                    n_hi = max(n_hi, children{q}.n_hi);

                    nsamp = nsamp + children{q}.n_samples;
                end

                %% ------------------------------------------------
                % Center of parent radar-data subdomain
                %
                % For receivers, choose receiver nearest the middle
                % of the block.
                %
                % For time, true midpoint can be between samples.
                %% ------------------------------------------------

                m_center = round(0.5 * (m_lo + m_hi));

                t_lo = (n_lo - 1) * tran_config.Ts;
                t_hi = (n_hi - 1) * tran_config.Ts;

                t_center = 0.5 * (t_lo + t_hi);

                %% ------------------------------------------------
                % Required parent image resolution
                %
                % Generalization of Boag's:
                %
                %   block data size 2^L x 2^L
                %            ->
                %   image size      2^L x 2^L
                %
                % Since our data are 40 x 1000 instead of NxN,
                % scale the image resolution approximately as
                %
                %       sqrt(samples contained / total samples).
                %
                % Root therefore has the requested final resolution.
                %% ------------------------------------------------

                fraction = nsamp / total_samples;

                nx_parent = ceil(Nx_final * sqrt(fraction));
                ny_parent = ceil(Ny_final * sqrt(fraction));

                nx_parent = max(nx_parent, 1);
                ny_parent = max(ny_parent, 1);

                nx_parent = min(nx_parent, Nx_final);
                ny_parent = min(ny_parent, Ny_final);

                % Root must be exactly final grid.
                if nsamp == total_samples
                    nx_parent = Nx_final;
                    ny_parent = Ny_final;
                end

                xp = linspace(x_lim(1), x_lim(2), nx_parent);
                yp = linspace(y_lim(1), y_lim(2), ny_parent);

                [Xp, Yp] = meshgrid(xp, yp);

                %% ------------------------------------------------
                % Parent phase center
                %% ------------------------------------------------

                phi_parent = center_phase( ...
                    Xp, Yp, z0, ...
                    rx_pos(m_center,:), ...
                    tx_pos, ...
                    t_center, ...
                    tran_config);

                %% ------------------------------------------------
                % Aggregate children
                %% ------------------------------------------------

                parent_img = complex(zeros(ny_parent, nx_parent));

                for q = 1:length(children)

                    child = children{q};

                    [ny_child, nx_child] = size(child.img);

                    xc = linspace(x_lim(1), x_lim(2), nx_child);
                    yc = linspace(y_lim(1), y_lim(2), ny_child);

                    %% Interpolate child's smooth phase-centered image
                    child_interp = interpolate_complex( ...
                        child.img, ...
                        xc, yc, ...
                        xp, yp, ...
                        opts.interp_method);

                    %% Child phase center evaluated on PARENT grid
                    phi_child = center_phase( ...
                        Xp, Yp, z0, ...
                        rx_pos(child.m_center,:), ...
                        tx_pos, ...
                        child.t_center, ...
                        tran_config);

                    %% --------------------------------------------
                    % Boag phase correction
                    %
                    % child:
                    %
                    % I~_c = exp(-j phi_c) I_c
                    %
                    % parent requires:
                    %
                    % exp(-j phi_p) I_c
                    %
                    % hence
                    %
                    % exp(j(phi_c - phi_p)) I~_c
                    %% --------------------------------------------

                    correction = exp( ...
                        1j * (phi_child - phi_parent));

                    parent_img = parent_img + ...
                        correction .* child_interp;

                end

                %% ------------------------------------------------
                % Store parent
                %% ------------------------------------------------

                parent.m_lo = m_lo;
                parent.m_hi = m_hi;

                parent.n_lo = n_lo;
                parent.n_hi = n_hi;

                parent.m_center = m_center;
                parent.t_center = t_center;

                parent.n_samples = nsamp;

                parent.img = parent_img;

                new_nodes{pm,pn} = parent;

            end
        end

        elapsed = toc;

        nodes = new_nodes;

        level_stats(level).time = elapsed; %#ok<AGROW>
        level_stats(level).num_nodes = Nm_new * Nn_new;

        if opts.verbose
            fprintf('  Level time: %.3f s\n', elapsed);
        end

    end

    %% ------------------------------------------------------------
    % Root node
    %% ------------------------------------------------------------

    root = nodes{1};

    % Its centered representation is:
    %
    %       I~ = exp(-j phi_root) I
    %
    % Recover complex matched-filter image.

    [X, Y] = meshgrid(x_vec, y_vec);

    phi_root = center_phase( ...
        X, Y, z0, ...
        rx_pos(root.m_center,:), ...
        tx_pos, ...
        root.t_center, ...
        tran_config);

    mf_img = exp(1j * phi_root) .* root.img;

    %% Information
    info.levels = level;
    info.level_stats = level_stats;
    info.total_samples = total_samples;
    info.final_image_size = [Ny_final, Nx_final];

end


%% ========================================================================
function phi = center_phase( ...
        X, Y, Z, rx, tx, t, cfg)
%CENTER_PHASE
%
% Matched-filter phase for the center of one radar-data block.
%
% Simulator:
%
% y = exp(-j 2*pi*fc*tau)
%     exp(+j pi*a*tau^2)
%     exp(-j 2*pi*a*tau*t)
%
% Therefore MF conjugate phase is
%
% phi = +2*pi*fc*tau
%       -pi*a*tau^2
%       +2*pi*a*tau*t

    d_tx = sqrt( ...
        (X - tx(1)).^2 + ...
        (Y - tx(2)).^2 + ...
        (Z - tx(3)).^2);

    d_rx = sqrt( ...
        (X - rx(1)).^2 + ...
        (Y - rx(2)).^2 + ...
        (Z - rx(3)).^2);

    tau = (d_tx + d_rx) / cfg.c;

    phi = ...
          2*pi*cfg.fc .* tau ...
        - pi*cfg.a .* tau.^2 ...
        + 2*pi*cfg.a .* tau .* t;

end


%% ========================================================================
function out = interpolate_complex( ...
        img, x_old, y_old, x_new, y_new, method)
%INTERPOLATE_COMPLEX
%
% Interpolate real and imaginary parts separately.

    [ny, nx] = size(img);

    %% One-pixel low-resolution image
    if nx == 1 && ny == 1
        out = img(1) * ...
            ones(length(y_new), length(x_new));
        return;
    end

    %% Degenerate x dimension
    if nx == 1

        re = interp1( ...
            y_old, real(img(:,1)), ...
            y_new, method, 'extrap');

        im = interp1( ...
            y_old, imag(img(:,1)), ...
            y_new, method, 'extrap');

        temp = complex(re, im).';

        out = repmat(temp, 1, length(x_new));
        return;
    end

    %% Degenerate y dimension
    if ny == 1

        re = interp1( ...
            x_old, real(img(1,:)), ...
            x_new, method, 'extrap');

        im = interp1( ...
            x_old, imag(img(1,:)), ...
            x_new, method, 'extrap');

        temp = complex(re, im);

        out = repmat(temp, length(y_new), 1);
        return;
    end

    %% Normal 2-D interpolation

    [Xold, Yold] = meshgrid(x_old, y_old);
    [Xnew, Ynew] = meshgrid(x_new, y_new);

    re = interp2( ...
        Xold, Yold, real(img), ...
        Xnew, Ynew, method);

    im = interp2( ...
        Xold, Yold, imag(img), ...
        Xnew, Ynew, method);

    out = complex(re, im);

end