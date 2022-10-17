% Function: calculate 2nd time series features
function [E, ro, F, LH] = compute_2nd_features_single(device)
    % Co-occurrence matrix
    % Q is the quantize level, here we choose it to be 2.

    Q = length(60:2:120)+1;
    % heart rate is quantized into Q levels.
    partition = 60:2:120;
    quat_level = 1:length(60:2:120)+1;
    
    %device = importdata(device_dir);
    
    [index, quat] = quantiz(device, partition, quat_level);
    c = zeros(length(quat_level));
    
    for i = 1:length(device)-1
        if (~isnan(device(i))) && (~isnan(device(i+1)))
        c(quat(i),quat(i+1)) = c(quat(i),quat(i+1)) + 1;
        end
    end
    
    c=c/sum(c,'all');
    
    % Energy
    c_e = c*1000;
    E = int16(sum(sum(c_e.^2))/length(device));

    % Entropy
    
    mu_x = sum((1:Q)'.*sum(c,2))/Q;
    mu_y = sum((1:Q).*sum(c,1))/Q;

    sigma_x = sqrt(sum(((1:Q)'-mu_x).^2.*sum(c,2))/Q);
    sigma_y = sqrt(sum(((1:Q)-mu_y).^2.*sum(c,1))/Q);

    % Correlation
    ro_xy = 0;
    for i = 1:Q
        for j = 1:Q
            ro_xy = ro_xy + (i-mu_x)*(j-mu_y)*c(i,j);
        end
    end
    
    ro = ro_xy / (sigma_x * sigma_y);

    % Inertia
    F = 0;
    for i = 1:Q
        for j = 1:Q
            F = F + (i-j)^2*c(i,j);
        end
    end

    % Local homogeneity
    LH = 0;
    for i = 1:Q
        for j = 1:Q
            LH = LH + c(i,j)/(1+(i-j)^2);
        end
    end
end