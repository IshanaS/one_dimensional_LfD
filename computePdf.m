function p = computePdf(obs,support,kernName,kernParam)

switch kernName
    case 'Gaussian'
        %kernParam is the covariance matrix
        if size(obs,2) > 1
            %p = mvnpdf(obs, support, kernParam);
            N = length(kernParam);
            p = 1;
            for n = 1 : N
                temp =  1/sqrt(2*pi*kernParam(n,n))*exp(-(obs(n) - support(n))^2/(2*kernParam(n,n)));
                p = p*temp;
            end
        else
            p = 1/sqrt(2*pi*kernParam)*exp(-(obs-support)^2/ (2*kernParam));
        end
        
    case 'Laplace'
        %kernParam is the diag matrix of elementwise standard deviation
        p = lappdf(obs, support, kernParam);
        
    case 'Rect'
        %kernParam is the diag matrix of window widths
        p = rectpdf(obs, support, kernParam);
end