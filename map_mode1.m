function mode_suggestion = map_mode1(new_measurement,M)
%  the inputs to be given are:
%     new_measurement = [x y theta];
%     M = mode implemented at the last time step
%     
    global kerName N Q mag classes P_modes trainModeDensity trainStateDensity feature particles_x l f_next;
    %recieve new measurement data - y(t)

    % convert to a feature vector (measured)
    feature = [new_measurement feature(1)];
    m_next = M*ones(N,1);% all particles now have the same mode which was implemented at the last time step
    
    % CORRECT the predicted distribution of x(t),m(t) which we had
    % obtained at last time step through propogation 
    %% assign weights to the existing points

    pt_wt = zeros(N,1);
    cul_wt = 0*pt_wt; % cumulative weight

    % weight of a particle = p(y|x)/normalizing factor
    for n = 1:N
        % measurement model is a sum of gaussians at 0 and the true value
        % since the true value is only guessed at 
        % all particle_x are considered true values 
        pt_wt(n) = P_measure_model(feature(1),f_next(n,1));%*p_mx(n);
        % the combined feature with old and new values 
        %cul_wt(n) = sum(pt_wt(1:n));
    end
    pt_wt = pt_wt/sum(pt_wt);% this is the normalizing factor
    %cul_wt = cul_wt/cul_wt(end);% this is the normalizing factor
    cul_wt = cumsum(pt_wt);
    
    %% resample and get new points with equal weights

    % determine N probabilities
    seed =  0 + (1/N - 0)*rand(1,1);
    p_seeds = [seed];
    while max(p_seeds) < 1-1/N
        p_seeds = [p_seeds;p_seeds(end)+1/N];
    end

    % determine the number of samples corresponding to the new probabilities
    count = 0*pt_wt;
    count(1) = sum(p_seeds<=cul_wt(1));
    for n = 2:N
        count(n) = sum(p_seeds<=cul_wt(n))-sum(count(1:n-1));
    end

    % get the vector of sample values (both positions and modes) corresponding to the new probabilities
    if max(count) > 0
        f_resampled = [];
        m_resampled = [];
        for n = 1:N
            if count(n)~=0
                f_resampled = [f_resampled; ones(count(n),1)*f_next(n,:)];
                m_resampled = [m_resampled; ones(count(n),1)*m_next(n)];
            end
        end
        particles_x = f_resampled(:,1);% corrected particle positions
        particles_m = m_resampled;% corrected mode values - same as original values since all particles were assigned mode M at the beginning
    end
    %% Propagation - prediction of x(t+1),m(t+1)

    % initialize the distribution (x and m) for t+1
    m_next = zeros(N,1);% initialize the modes of particles
    f_next = zeros(N,2);% initialize positions/states of particles [x+ x]
    p_mx = zeros(N,1);% initialize the importance of particles

    % at each x_i evaluate a possible x_i(t+1)
    
    for n = 1:N % iteration over particles

        % the guess of the true value of this particle
        f = particles_x(n,:);
        m = particles_m(n);

        %importance sampling

        % to choose next m - generate Q possiblities using uniform dist
        m_uniform_sample_ind = ceil(length(classes)*rand(Q,1));
        m_uniform_samples = classes(m_uniform_sample_ind);

        % initialize the probability of each of them
        m_uniform_samples_p = zeros(size(m_uniform_samples));
        
        % P(m+1|x,m) = P(x|m,m+1)*P(m+1|m)/sum(P(x|m,m+1)) over all m+1

        % evaluation the P(x|m,m+1) for all m+1 
        % assume m as fixed since we already have information from the prev step
        All_pm = zeros(max(classes),1);% 2x1 array
        for m2 = 1:size(classes)% index of each class
            m1 = find(classes==m);% the m(t) = m1
            All_pm(classes(m2)) = ObservationLikelihood(f, trainModeDensity(m1,m2).Xsv, trainModeDensity(m1,m2).Zsv, kerName, l*eye(1))*P_modes(m1,m2);
        end

        if sum(All_pm)~= 0
            All_pm = All_pm/sum(All_pm);
        end

        % evaluate the probability of the new mode samples - P(m+1|x,m)
        for i = 1:length(m_uniform_samples_p)
            m_uniform_samples_p(i) = All_pm(m_uniform_samples(i));
        end

        % evaluate importance at each of them
        m_uniform_samples_p = m_uniform_samples_p*Q;


        % to choose next feature - generate Q possiblities using uniform dist
        f_uniform_samples = [ones(Q,1)*f + mag*(rand(Q,1)-0.5) ones(Q,1)*f];% [x+ x] 1x2 vector each Q times

        % evaluate the probability of each of them

        f_uniform_samples_p = 0*m_uniform_samples;% initialize
        
        % P([x+ x]|m)
        for i = 1:size(f_uniform_samples_p,1)
            f_uniform_samples_p(i) = ObservationLikelihood(f_uniform_samples(i,:), trainStateDensity(m).Num_Xsv, trainStateDensity(m).Num_Zsv, kerName, l*eye(2));
        end
        % P(x|m)
        f_uniform_samples_p_denom = ObservationLikelihood(f, trainStateDensity(m).Num_Xsv, trainStateDensity(m).Num_Zsv(:,2), kerName, l*eye(1));

        % ensure non zero denominator 
        if f_uniform_samples_p_denom ~= 0
            f_uniform_samples_p = f_uniform_samples_p/f_uniform_samples_p_denom;
        else
            f_uniform_samples_p = f_uniform_samples_p/max(max(f_uniform_samples_p),10^-7);
        end

        % evaluate importance at each of them
        f_uniform_samples_p = f_uniform_samples_p*Q;

        % pick the point with the max importance or p_m*p_x;

        % evaluate p_m*p_x for all points in the importance sampling step
        p_mx_IS = m_uniform_samples_p.*f_uniform_samples_p/(Q^2);

        [p_mx(n),ind_mf] = max(p_mx_IS);

        f_next(n,:) = f_uniform_samples(ind_mf,:);% next feature(new & old position) of the particles [x+ x] predicted
        m_next(n) = m_uniform_samples(ind_mf);% next mode of the particles m+ predicted
    end

    mode_suggestion = m_next;
end