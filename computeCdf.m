function p = computeCdf(obs,support,kerName,kernParam)

switch kerName
    case 'Gaussian'
        %kernParam is the covariance matrix
        if size(obs,2) > 1
            %p = mvncdf(obs, support, kernParam);
            N = length(kernParam);
            p = 1;
            for n = 1 : N
                temp =  0.5*erfc(-(obs(n) - support(n))/sqrt(2*kernParam(n,n)));
                p = p*temp;
            end
        else
            p = 0.5*erfc(-(obs - support)/sqrt(2*kernParam));
        end
        
    case 'Laplace'
        %kernParam is the diag matrix of elementwise standard deviation
        p = lapcdf(obs, support, kernParam);
        
    case 'Rect'
        %kernParam is the diag matrix of window widths
        p = rectcdf(obs, support, kernParam);
end