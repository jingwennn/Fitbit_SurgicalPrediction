% Function: compute DFA for sleep
function [F_n, skewness_out, kurtosis_out] = dfa_sleep_single(device)
    %device = importdata(device_dir);
    % try n windows 
    n = 10:10:60;
    N1 = length(n);
    F_n = zeros(N1,1);
    for i=1:N1
        F_n(i) = DFA(device,n(i),1);
    end
    skewness_out = skewness(device);
    kurtosis_out = kurtosis(device);
end