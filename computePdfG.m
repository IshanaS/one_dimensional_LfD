function p = computePdfG(obs,support,kernParam)

%kernParam is the covariance matrix

N = length(kernParam);
ss = 0*obs(:,1);
for n = 1 : N
    ss =  ss + ((obs(:,n) - support(n)).^2 );
end

%ss = sum((obs - repmat(support, size(obs, 1), 1)).^2, 2);
p = exp(-0.5* ss/ kernParam(1,1))/ (sqrt(2*pi* kernParam(1,1)))^N;
%p = exp(-0.5* ss/ kernParam(1,1))/ (sqrt(2*pi* kernParam(1,1)))^N;