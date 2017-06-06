function p = P_measure_model(measurement,true_val)
    % for 1D
    p = 0.05*mvnpdf(measurement,zeros(size(true_val)),0.001) + 0.95*mvnpdf(measurement,true_val,diag([0.05]));
%     % for hexapod
%     p = 0.05*mvnpdf(measurement,zeros(size(true_val)),0.001*eye(3)) + 0.95*mvnpdf(measurement,true_val,diag([0.05,0.05,0.05]));
end