function [targets, K] = sample_targets_simple(r_min, r_max, z_min, z_max, max_targets)
    %SAMPLE_TARGETS_BY_QUADRANTS
    % Randomly samples 1 to max_targets targets.
    %
    % Outputs:
    %   targets   - K x 3 matrix, each row is [x, y, z]
    %   quadrants - K x 1 vector of selected quadrants

    % 1. Randomly choose number of targets: K in {1, ..., max_targets}
    K = randi(max_targets);

    % Allocate target matrix
    targets = zeros(max_targets, 3);

    for i = 1:K
        u = rand();
        r = sqrt(u * (r_max^2 - r_min^2) + r_min^2);

        theta = rand_uniform(0, 2*pi);
        
        x = r * cos(theta);
        y = r * sin(theta);
        z = z_min + (z_max - z_min) * rand(1, 1);

        targets(i, :) = [x, y, z];
    end
end

function val = rand_uniform(a, b)
%RAND_UNIFORM Samples uniformly from [a, b]
val = a + (b - a) * rand();
end