function p = ObservationLikelihood(obs, Xsv, Zsv, kerName, kernParam)
%OBSERVATIONLIKELIHOOD To compute likelihood of a several observations
%using the supoort vector density estimate
% Input: 
%   obs: Data (each row is an observation)
%   Zsv: Data points selected as support vectors
%   Xsv: Prior probability associated with each support vector
%   kernName: Name of the kernel to be used (currently supports 'Gaussian'
%   and 'Laplace')
%   kernParam: kernel parameter (covariance matrix for Gaussian or diagonal
%   standard deviation matrix for Laplace)
% Output:
%   p: Measurement likelihood of each measurement in 'obs'
% Author: Nurali Virani (nnv105@psu.edu)
% Last edited: 11/24/2014

p = zeros(size(obs,1),1);
% for i = 1 : size(obs,1)
    for j = 1 : length(Xsv)
%         p(i) = p(i) + Xsv(j)*computePdf(obs(i,:),Zsv(j,:),kerName,kernParam);
        p = p + Xsv(j)*computePdfG(obs,Zsv(j,:),kernParam);
    end 
% end
    
end

