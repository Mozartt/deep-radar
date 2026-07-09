function [targets, K] = sample_targets_by_quadrants(r_min, r_max, z_min, z_max, max_targets)
    %SAMPLE_TARGETS_BY_QUADRANTS
    % Randomly samples 1 to max_targets targets.
    % Each target is placed in a different quadrant.
    % Quadrants are sampled without replacement.
    %
    % Outputs:
    %   targets   - K x 3 matrix, each row is [x, y, z]
    %   quadrants - K x 1 vector of selected quadrants

    % 1. Randomly choose number of targets: K in {1, ..., max_targets}
    K = randi(max_targets);

    all_quadrants = randperm(4);

    quadrants = all_quadrants(1:K).';

    % Allocate target matrix
    targets = zeros(max_targets, 3);

    for i = 1:K
        q = quadrants(i);
        % 3. Sample radius uniformly in area
        % This gives uniform spatial density in the disk/annulus.
        u = rand();
        r = sqrt(u * (r_max^2 - r_min^2) + r_min^2);

         % 4. Sample theta uniformly inside the selected quadrant
        switch q
            case 1
                theta = rand_uniform(0, pi/2);

            case 2
                theta = rand_uniform(pi/2, pi);

            case 3
                theta = rand_uniform(pi, 3*pi/2);

            case 4
                theta = rand_uniform(3*pi/2, 2*pi);

            otherwise
                error('Quadrant must be 1, 2, 3, or 4.');
        end

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