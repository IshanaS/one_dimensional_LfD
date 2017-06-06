function [Zsv, Xsv, optout] = SupportVectorDensityEstimation(z, kernName, kernParam)
%[Zsv,Xsv]=SupportVectorDensityEstimation(z, kernName, kernParam)
%   To perform nonparametric density estimation for a dataset given in z,
%using a particular kernel.
% Input:
%   z: Data (each row is an observation)
%   kernName: Name of the kernel to be used (currently supports 'Gaussian',
%   'Laplace', and 'Rect')
%   kernParam: kernel parameter (covariance matrix for Gaussian or diagonal
%   standard deviation matrix for Laplace or diagonal matrix of half window
%   width for Rectangular kernels)
% Output :
%   Zsv: Data points selected as support vectors
%   Xsv: Prior probability associated with each support vector
%   optout: Optional outputs --
%           s1  : residuals over the margin negative side   
%           s2  : residuals over the margin positive side
%           x   : solution vector (size N)
%           si  : error margin
%           obj : Objective function value
%           tim : Execution time [for Y, for tildeK, for optimization]
% Example Usage:
%   Z = [randn(100,1);3+randn(100,1)];
%   [zSV, xSV, optout] = SupportVectorDensityEstimation(Z, 'Gaussian', 0.5);
%   scatter(Z,zeros(size(Z)));
%   hold on
%   scatter(zSV,zeros(size(zSV)),'rx');
% Author: Nurali Virani (nnv105@psu.edu)
% Last edited: (11/7/2016) 

%parpool

optout = struct();
s1 = [];
s2 = [];
% Measurement or feature dimensions
dim = size(z,2);
% Number of measurement samples
count = size(z,1);

% Optimization Parameters
%nu-SVDE
nuCVal = 100;  
nuVal = 0.01;

%e-SVDE
eCVal = 10^4; 
siVal = 1e-4;

%Choose solver and technique
solver = 11;

%Regression output for training
%Empirical distribution
display('Computing Empirical Distribution');
tic;

%Brute force computation
cdfval = zeros(count,1);
parfor i = 1:count
    sum1 = 0;
    for j = 1:count
       h = prod(z(i,:) >= z(j,:));
       sum1 = sum1 + h;
    end
    cdfval(i) = sum1/count;
end

tElap1 = toc;
display(['Time taken: ', num2str(tElap1)]);

%Cumulative Density Matrix and Gram Matrix computation
display('Computing Matrices used in Optimization');
tic;
cdMat = ones(count);
kMat = ones(count);

parfor i = 1:count
    for j = 1:count
            cdMat(i,j) = computeCdf(z(i,:),z(j,:),kernName,kernParam);
    end
end


% parfor i = 1:count
%     for j = 1:count
%             kMat(i,j) = computePdf(z(i,:),z(j,:),kernName,kernParam);
%     end
% end

tElap2 = toc;
display(['Time taken: ', num2str(tElap2)]);

%optimization problem
display('Solving Optimization Problem...');
tic;
switch solver
    case 1
        %nuSVDE as QP with CVX
        nu = nuVal;
        ep = 0.0001;
        C = nuCVal;
        cvx_begin
            cvx_precision best
            variables x(count) s1(count) s2(count) si;
            dual variables a1 a2 b1 b2 d l phi
            minimize(x'*(kMat+ep*eye(count))*x+ C*(sum(s1+s2)/count+nu*si));%kMat + ep*
            a1: cdfval - cdMat*x -s1<= si; %#ok<VUNUS>
            a2: -cdfval + cdMat*x -s2<= si; %#ok<VUNUS>
            l: sum(x) == 1; %#ok<EQEFF>
            d: x >= 0; %#ok<VUNUS>
            b1: s1 >= 0;
            b2: s2 >= 0;
            phi: si >= 0;
        cvx_end
        
        optout.s1 = s1;
        optout.s2 = s2;
        optout.si = si;
        optout.x = x; 
        optout.obj = cvx_obj;
    case 2
        %e-SVDE with CVX
        ep = 0.0001;
        si = siVal;
        cvx_begin
            cvx_precision high
            variables x(count)
            minimize(quad_form(x,kMat + ep*eye(count)))
            cdfval - cdMat*x <= si; %#ok<VUNUS>
            -cdfval + cdMat*x <= si; %#ok<VUNUS>
            sum(x) == 1; %#ok<EQEFF>
            x >= 0; %#ok<VUNUS>
        cvx_end
        
        optout.si = si;
        optout.x = x; 
        optout.obj = cvx_obj;
    case 3
        %nuSVDE with YALMIP
        nu = nuVal;
        ep = 0.01;
        C = nuCVal; 
        xhat = sdpvar(count,1);
        s1hat = sdpvar(count,1);
        s2hat = sdpvar(count,1);
        sihat = sdpvar(1,1);
        Constraints = [cdfval - cdMat*xhat - s1hat <= sihat, -cdfval + cdMat*xhat - s2hat <= sihat, sum(xhat) == 1, xhat >= 0, sihat >=0, s1hat >= 0, s2hat >= 0];
        Obj = xhat'*(kMat+ep*eye(count))*xhat+ C*(sum(s1hat+s2hat)/count+nu*sihat);
        optimize(Constraints, Obj);
        si = value(sihat);
        s1 = value(s1hat);
        s2 = value(s2hat);
        x = value(xhat);
        
        optout.s1 = s1;
        optout.s2 = s2;
        optout.si = si;
        optout.x = x; 
        optout.obj = value(Obj);
    case 4
        %e-SVDE with YALMIP
        si = siVal;
        ep = 0.01;
        xhat = sdpvar(count,1);
        Constraints = [cdfval - cdMat*xhat <= si, -cdfval + cdMat*xhat <= si, sum(xhat) == 1, xhat >= 0];
        Obj = xhat'*(kMat+ep*eye(count))*xhat;
        optimize(Constraints, Obj);
        x = value(xhat);
        
        optout.x = x; 
        optout.obj = value(Obj);
    case 5
        %e-SVDE with slacks and G = kMat using YALMIP
        si = siVal;
        ep = 1e-4;
        C = eCVal; 
        xhat = sdpvar(count,1);
        s1hat = sdpvar(count,1);
        s2hat = sdpvar(count,1);
        Constraints = [cdfval - cdMat*xhat - s1hat <= si, -cdfval + cdMat*xhat - s2hat <= si, sum(xhat) == 1, xhat >= 0, s1hat >= 0, s2hat >= 0];
        Obj = C*(sum(s1hat+s2hat)/count) + xhat'*(kMat+ep*eye(count))*xhat;
        options = sdpsettings('solver','mosek');
        options.mosek.MSK_IPAR_INTPNT_MAX_ITERATIONS = 1000;
        options.mosek.MSK_IPAR_NUM_THREADS = 6;
        optimize(Constraints, Obj, options);
        x = value(xhat);
        s1 = value(s1hat);
        s2 = value(s2hat);
        
        optout.s1 = s1;
        optout.s2 = s2;
        optout.x = x; 
        optout.obj = value(Obj);
     case 6
        %e-SVDE with slacks and G = 1 using YALMIP
        si = siVal;
        ep = 0.01;
        C = eCVal; 
        xhat = sdpvar(count,1);
        s1hat = sdpvar(count,1);
        s2hat = sdpvar(count,1);
        Constraints = [cdfval - cdMat*xhat - s1hat <= si, -cdfval + cdMat*xhat - s2hat <= si, sum(xhat) == 1, xhat >= 0, s1hat >= 0, s2hat >= 0];
        Obj = xhat'*xhat + C*(sum(s1hat+s2hat)/count);
        optimize(Constraints, Obj);
        x = value(xhat);
        s1 = value(s1hat);
        s2 = value(s2hat);
        
        optout.s1 = s1;
        optout.s2 = s2;
        optout.x = x; 
        optout.obj = value(Obj);
    case 7
        %nuSVDE as LP with CVX
        nu = nuVal;
        %cdMat1 = cdMat - 0.5*eye(size(cdMat)); 
        cvx_begin
            cvx_precision best
            variables x(count) s1(count) s2(count) si;
            dual variables a1 a2 b1 b2 l d phi;
            minimize(sum(s1+s2)/count + (nu*si));%
                a1 : cdfval - cdMat*x -s1<= si; %#ok<VUNUS>
                a2 : -cdfval + cdMat*x -s2<= si; %#ok<VUNUS>
                l : sum(x) == 1; %#ok<EQEFF>
                d : x >= 0; %#ok<VUNUS>
                b1 : s1 >= 0;%#ok<VUNUS>
                b2 : s2 >= 0;%#ok<VUNUS>
                phi : si >= 0;%#ok<VUNUS>
        cvx_end
        obj = cvx_optval;
        optout.s1 = s1;
        optout.s2 = s2;
        optout.x = x; 
        optout.obj = obj;
    case 8 
        %nuSVDE dual problem
        nu = nuVal;
        C = nuCVal;
        cvx_begin
            cvx_precision best
            variables a1(count) a2(count);
            minimize(1/2*(a1-a2)'*(kMat)*(a1-a2) - cdfval'*(a1-a2));
                a1 >= 0;
                a1 <= C;
                a2 >= 0;
                a2 <= C;
                a1 - a2 >= 0;
                sum(a1 - a2) == 1;
                sum(a1 + a2) <= C*nu*count;
        cvx_end
        x = a1 - a2;
        si = max(cdfval - cdMat*x);
        obj = cvx_optval;
        optout.x = x; 
        optout.obj = obj;
    case 9
        %nuSVDE as LP with CVX without slack
        %cdMat1 = cdMat - 0.5*eye(size(cdMat)); 
        cvx_begin
            cvx_precision best
            variables x(count) si;
            dual variables a1 a2 l d phi;
            minimize(si);%
                a1 : cdfval - cdMat*x <= si; %#ok<VUNUS>
                a2 : -cdfval + cdMat*x <= si; %#ok<VUNUS>
                l : sum(x) == 1; %#ok<EQEFF>
                d : x >= 0; %#ok<VUNUS>                
                phi : si >= 0;%#ok<VUNUS>
        cvx_end
        obj = cvx_optval;
        optout.x = x; 
        optout.obj = obj;
    case 10
        %LP for DE using linprog (MOSEK)
        nu = nuVal;
        
        %cost functional
        c = [zeros(count,1); nu; (1/count)*ones(2*count,1)];
        
        %inequality
        A = [-cdMat -ones(count,1) -eye(count) zeros(count); cdMat -ones(count,1) zeros(count) -eye(count)];
        b = [-cdfval; cdfval];
        
        %equality
        Aeq = [ones(1,count), zeros(1, 2*count+1)];
        beq = 1;
        
        %bounds;
        lb = zeros(3*count+1,1);
        %ub = zeros(3*count+1,1);
        
        %without MOSEK (use this one)
        %options = optimoptions(@linprog, 'Algorithm', 'interior-point', 'Display', 'iter', 'TolFun', 1e-10);
        
        %with MOSEK (use this one)
        options = mskoptimset('Display','iter','MaxIter',500);
        %'interior-point' 1e-15
        %options = optimoptions(@linprog, 'Algorithm', 'dual-simplex', 'Display', 'iter', 'TolFun', 1e-10);

        [xval,fval,exitflag,output,lambda] = linprog(c, A, b, Aeq, beq, lb, [], [], options);

        x = xval(1:count);
        si = xval(count + 1);
        s1 = xval(count + 2 : 2*count + 1);
        s2 = xval(2*count + 2 : 3*count + 1);
        obj = fval;
        optout.s1 = s1;
        optout.s2 = s2;
        optout.si = si;
        optout.x = x; 
        optout.obj = obj;
    case 11
        %LP for DE using mosekopt(MOSEK) (most optimized version)
        nu = 0.01; %nuVal;
 
        %         %specify Q
        %         klt = tril(kMat);
        %         [i1,j1,v1] = find(klt);
        % 
        %         prob.qosubi = [i1; [count+1 : 3*count+1]']; 
        %         prob.qosubj = [j1; [count+1 : 3*count+1]']; 
        %         prob.qoval  = [v1; ep*ones(2*count + 1,1)];
     
        % Specify the c vector. 
        problem.c = [zeros(count,1); nu; (1/count)*ones(2*count,1)]; 

        %inequality
        A = [-cdMat -ones(count,1) -eye(count) zeros(count); 
              cdMat -ones(count,1) zeros(count) -eye(count);
               ones(1,count), zeros(1, 2*count+1);
               -ones(1,count), zeros(1, 2*count+1)];
        b = [-cdfval; cdfval; 1; -1];
        
        % Specify a in sparse format. 
        problem.a = sparse(A); 

        % Specify lower bounds of the constraints. 
        problem.blc  = []; 

        % Specify  upper bounds of the constraints. 
        problem.buc  = b; 

        % Specify lower bounds of the variables. 
        problem.blx  = zeros(3*count + 1, 1); 

        % Specify upper bounds of the variables. 
        problem.bux = [];% [Inf*ones(count,1); 0.01/count; Inf*ones(2*count,1)];   % There are no bounds. 

        [~,ar]=mosekopt('symbcon echo(0)');
        optimparam = ar.symbcon;

        % Relative primal-dual gap tolerance. 
        myparam.MSK_DPAR_INTPNT_TOL_REL_GAP = 1.0e-8; 
        myparam.MSK_DPAR_INTPNT_TOL_DFEAS = 1.0e-8; 
        myparam.MSK_DPAR_INTPNT_TOL_PFEAS = 1.0e-8; 
        myparam.MSK_IPAR_INTPNT_BASIS = optimparam.MSK_BI_NEVER;
        myparam.MSK_IPAR_NUM_THREADS = 6;
        myparam.MSK_IPAR_INTPNT_MAX_ITERATIONS = 30;
        myparam.MSK_IPAR_INTPNT_SOLVE_FORM = optimparam.MSK_SOLVE_DUAL;
         
        % Perform the optimization. 
        [r,res] = mosekopt('minimize echo(0)',problem,myparam); 
        xval = res.sol.itr.xx;
        x = xval(1:count);
        si = xval(count + 1);
        s1 = xval(count + 2 : 2*count + 1);
        s2 = xval(2*count + 2 : 3*count + 1);   
        obj = res.sol.itr.pobjval;
        
        optout.s1 = s1;
        optout.s2 = s2;
        optout.si = si;
        optout.x = x; 
        optout.obj = obj;
    case 12
        %LP for DE using linprog (MOSEK) without slack
        nu = [];
        
        %cost functional
        c = [zeros(count,1); 1];
        
        %inequality
        A = [-cdMat -ones(count,1); cdMat -ones(count,1)];
        b = [-cdfval; cdfval];
        
        %equality
        Aeq = [ones(1,count), zeros(1)];
        beq = 1;
        
        %bounds;
        lb = zeros(count+1,1);
        %ub = zeros(3*count+1,1);
        
        options = optimoptions(@linprog, 'Algorithm', 'interior-point', 'Display', 'iter', 'TolFun', 1e-10);
        %'interior-point' 1e-15
        %options = optimoptions(@linprog, 'Algorithm', 'dual-simplex', 'Display', 'iter', 'TolFun', 1e-10);

        [xval,fval,exitflag,output,lambda] = linprog(c, A, b, Aeq, beq, lb, [], []);

        x = xval(1:count);
        si = xval(count + 1);
        obj = c'*xval;
        optout.si = si;
        optout.x = x; 
        optout.obj = obj;
end
tElap3 = toc;
display(['Time taken: ', num2str(tElap3)]);

display(strcat('Nu : ',num2str(nu)));

%Check the value of insensitivity parameter
disp(strcat('Value of insensitivity parameter: ',num2str(si)));

%Find support vectors
sv = find(x>1e-5);
display(strcat('Number of components: ',num2str(length(sv))));

%Assign return values
Zsv = z(sv,:);
Xsv = x(sv);
optout.tim = [tElap1, tElap2, tElap3];

end


