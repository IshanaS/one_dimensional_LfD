% implement the particle filter for a 1 D case. 
clear all
clc
close all

% 
rep_density = 1;

% stop and record data at predetermined slow points (say)

% this pre-determined slow points aren't given to us as such and we have to
% learn them from the previous runs

global kerName l N Q mag classes P_modes trainModeDensity trainStateDensity feature particles_x f_next;

classes = [1;2];
num_of_exp = 10;
num_data_pts = 100;

%% step 1 - create the taining data from a model

% The data generation will determine the possible stop points using a
% gaussian across the ideal stop/slow points. Let us do this example for 2 
% points. 

% mode 1 : high speed - 5 mag/step
% mode 2 : low speed - 1 mag/step

x_measurement = zeros(50,100);
m_measurement = zeros(50,100);
m_measurement(:,1) = ones(size(m_measurement,1),1);
for exp = 1:num_of_exp
    % initialization at the beginning of experiment
    v = [5,1]; % velcoities at the 2 modes
    T = 0.05;% sec time step
    noise_mag = 0.001;% noise in measurement 

    for i = 2:num_data_pts
        x0 = x_measurement(exp,i-1);
        
        % choose the next mode
        m = normpdf(x0,2,1) + normpdf(x0,6,1);
        m_measurement(exp,i) = (m>0.3) + 1;
        
        % change the next state
        x_measurement(exp,i) = x0 + v(m_measurement(exp,i))*T + noise_mag*randn();
    end
    
end

% experiment repeated X (50 here) number of times and 
% each run yeilds Y (100 here) number of measurements of x
% corresponding to each x is the "label" or the mode

% let us say the stop point is at 2 and 6. Then the distribution of mode=2 
% is focussed around those points.

%visualize
figure(1);
for a = 1:size(x_measurement,1)
    scatter(x_measurement(a,:),m_measurement(a,:),'k');
    hold on;
end
title('Generated data - used for training')
xlabel('state');
ylabel('mode');

%% step 2 - get data in [x(t+1) x(t) m(t) m(t+1)] form

x_cur = [];
x_next = [];
m_cur = [];
m_next = [];

for b = 1:num_of_exp
    x_cur = [x_cur;x_measurement(b,1:end-1)'];
    x_next = [x_next;x_measurement(b,2:end)'];
    m_cur = [m_cur; m_measurement(b,1:end-1)'];
    m_next = [m_next; m_measurement(b,2:end)' ];
end
data = [x_next x_cur m_cur m_next];

%% step 3 - prepare datasets for density training

P_modes = zeros(length(classes));

for i = 1:length(classes) % prior mode
    m = classes(i);
    
    % Feature_data has all the data with current mode = i
    Feature_data = data(data(:,3)==i,:);

    % state dynamics - P(x(t+1)|x(t),m(t))
    trainStateDensity(i).data = Feature_data;
    trainStateDensity(i).Num_z = Feature_data(:,1:2);% P(x(t+1),x(t)|m(t))
    trainStateDensity(i).Denom_Z = Feature_data(:,2);% P(x(t)|m(t))
    
    % mode switching dynamics - P(x(t)|m(t+1),m(t))*P(m(t+1)|m(t))
    for j = 1:length(classes) % next mode
        trainModeDensity(i,j).data_z = Feature_data(Feature_data(:,4)==j,2);
        trainModeDensity(i,j).count = size(trainModeDensity(i,j).data_z,1);
        P_modes(i,j) = size(trainModeDensity(i,j).data_z,1);
    end
end

for c = 1:size(P_modes,1)
    P_modes(c,:) = P_modes(c,:)/sum(P_modes(c,:));
end

%% Step 4 - train the datasets to get densities of state and mode dynamics

kerName = 'Gaussian';
l = 0.0005;
%learn the densities
for i = 1:length(classes)
    [Num_Zsv, Num_Xsv, ~] = SupportVectorDensityEstimation(trainStateDensity(i).Num_z, kerName, l*eye(2));
    trainStateDensity(i).Num_Zsv = Num_Zsv;
    trainStateDensity(i).Num_Xsv = Num_Xsv;
    trainStateDensity(i).Denom_Zsv = Num_Zsv(:,2);% change in the hexapod
    trainStateDensity(i).Denom_Xsv = Num_Xsv;
    for j = 1:length(classes)
        [Zsv, Xsv, ~] = SupportVectorDensityEstimation(trainModeDensity(i,j).data_z, kerName, l*eye(1));
        trainModeDensity(i,j).Zsv = Zsv;
        trainModeDensity(i,j).Xsv = Xsv;
    end
end

%% step 5 - create a distribution of likelihood of the next states over the entire state space

D_dot = [];

% prepare a dataset of current [x(t),m(t)] to be used as starting point
if rep_density == 1
    test_pt_x = [1,2,3];
    start_x = test_pt_x;
    start_m = normpdf(start_x,2,1) + normpdf(start_x,6,1);
    start_m = (start_m > 0.3) + 1;
%     start_m(1,:) = ones(size(test_pt_x));
%     start_m(2,:) = 2*ones(size(test_pt_x));
elseif rep_density == 0
    start_x = linspace(0,10,30);
    % choose the mode
    start_m = normpdf(start_x,2,1) + normpdf(start_x,6,1);
    start_m = (start_m > 0.3) + 1; 
end

if rep_density == 0
    figure(3);
    scatter(start_x,start_m);
end

if rep_density == 1
    figure()
end

% prepare a dataset of possible states over the state space
next_x = linspace(0.5,3.5,402);
next_x = next_x(2:end-1);

% for each point from the start_x, test over a grid of possible next points

% initialize probability of the next state
p_x = zeros(length(next_x),1);
P = zeros(length(next_x),4);

% for each start point - get the next state likelihoods
for d = 1:length(start_x) % each start x
    for g = 1:size(start_m,1) % corresponding start m
        m = start_m(g,d); % current mode (m)
        x_cur = start_x(d); % current state (x)

        for k = 1:length(next_x) % get the likelihood of each next state possiblity

            x_next = next_x(k);% choose a next state to evaluate at
            % P([x+ x]|m)
            p_xN = ObservationLikelihood([x_next x_cur], trainStateDensity(m).Num_Xsv, trainStateDensity(m).Num_Zsv, kerName, l*eye(2));
            % P(x|m)
            p_xD = ObservationLikelihood(x_cur, trainStateDensity(m).Num_Xsv, trainStateDensity(m).Num_Zsv(:,2), kerName, l*eye(1));

            % to avoid dividing by 0 
            if p_xD ~= 0
                p_x(k) = p_xN/p_xD;
            else
                p_x(k) = p_xN/max(max(p_xN),10^-47);
            end

            % record the num and denom for debugging purposes
            P(k,1) = p_xN;
            P(k,2) = p_xD;
            P(k,3) = p_xN/p_xD;
        end

        % normalizing to get the probability instead of the likelihood
        P(:,4) = P(:,3)/sum(P(:,3));

        if rep_density == 1
            if m == 1
                h(1) = plot(next_x,P(:,4),'k','DisplayName','$m_{t} = 1$','LineWidth',3);
            elseif m == 2
                h(2) = plot(next_x,P(:,4),'--k','DisplayName','$m_t = 2$','LineWidth',3);
            end
            hold on;
            title('$\mathbf{P(x_{t+1} |x_{t},m_{t})}$','Interpreter','latex','FontSize',20);
            xlabel('$\mathbf{x}$','Interpreter','latex','FontSize',16)
            ylabel('$\mathbf{P(x_{t+1} |x_{t},m_{t})}$','Interpreter','latex','FontSize',16)
            end

        if rep_density == 0
            
            figure(2)
            scatter(x_cur,0,'filled','k');
            hold on;
            plot(next_x,P(:,4));
            hold on;
            
            [~,ind1] = max(P(:,4));
            figure(1);
            scatter(x_cur,0,'filled','k');
            hold on;
            scatter(next_x(ind1),0,'filled');
            hold on;

            D_dot = [D_dot;(next_x(ind1)-x_cur)/T];
            figure(5);
            scatter(x_cur,D_dot(d),'k')
            hold on;
            scatter(x_cur,v(m),'filled','k')

            figure(4);
            plot(next_x,log(P(:,3)));
            hold on;
        end
    end
end

if rep_density == 1 & g == 1
    h(3) = scatter(start_x,0*start_x,100,'filled','k','DisplayName','$x_{t}$');set(gca,'FontSize',12);
    lg1 = legend(h(1:3));
    lg1.FontSize = 12;
    lg1.Interpreter = 'latex';
    lg1.Location = 'northwest';
end
if rep_density == 0
    figure(2);
    title('Probability of possible next states');
    ylabel('P(x_{t+1})');
    
    figure(4);
    title('Log-likelihood of possible next states');
    ylabel('L(x_{t+1})');

    figure(1);
    title('Most likely next state + known modes');

    figure(5);
    title('Velocity predicted at the current x')
    xlabel('x_{t}')
    ylabel('V')
end

%% step 6 - create a distribution of likelihood of the next modes over the entire state space & mode space

% Evaluate P(m+1|x,m) over all combinations of x and m

% P(m+1|x,m) = P(x|m,m+1)*P(m+1|m)/sum(P(x|m,m+1)*P(m+1|m)) over all m+1
% for given x

% prepare a dataset of current [x(t),m(t)] to be used as starting point
if rep_density == 1
    clear test_pt_x
    test_pt_x = linspace(0,8,24);%[1,2,3,4,5,6,7,8];
    start_x = test_pt_x;
    clear start_m
    start_m(1,:) = ones(size(test_pt_x));
    start_m(2,:) = 2*ones(size(test_pt_x)); 
end

for g = 1:size(start_m,1)% corresponding start m
     p_mm = [];
     for d = 1:length(start_x) % each start x
        m = start_m(g,d); % current mode (m)
        x_cur = start_x(d); % current state (x)
        
        % evaluation the P(x|m,m+1) for all m+1 
        % assume m as fixed since we already have information from the prev step
        All_pm = zeros(max(classes),1);% 2x1 array
        for m2 = 1:size(classes)% index of each class
            m1 = find(classes==m);% the m(t) = m1
            All_pm(classes(m2)) = ObservationLikelihood(x_cur, trainModeDensity(m1,m2).Xsv, trainModeDensity(m1,m2).Zsv, kerName, l*eye(1))*P_modes(m1,m2);
        end

        if sum(All_pm)~= 0
            All_pm = All_pm/sum(All_pm);
        end
        a = [];
        for m2 = 1:size(classes)
            a = [a;All_pm(m2)];
        end
        p_mm = [p_mm a];
        % evaluate the probability of the new mode samples - P(m+1|x,m)
        
     end
    p_m(g).data = p_mm;
end

figure();
subplot(2,1,1);
plot(test_pt_x,p_m(1).data(1,:),'--k','LineWidth',3);set(gca,'FontSize',12);
hold on
plot(test_pt_x,p_m(1).data(end,:),'k','LineWidth',3);set(gca,'FontSize',12);
title('$\mathbf{m_t = 1}$','Interpreter','latex','FontSize',20);
lg2 = legend('$m_{t+1} = 1$','$m_{t+1} = 2$');
lg2.Interpreter = 'latex';
lg2.FontSize = 12;
xlabel('$\mathbf{x_{t}}$','Interpreter','latex','FontSize',16)
ylabel('$\mathbf{P(m_{t+1}|x_{t},m_{t})}$','Interpreter','latex','FontSize',16)
subplot(2,1,2);
plot(test_pt_x,p_m(2).data(1,:),'--k','LineWidth',3);set(gca,'FontSize',12);
hold on;
plot(test_pt_x,p_m(2).data(end,:),'k','LineWidth',3);set(gca,'FontSize',12);
title('$\mathbf{m_t = 2}$','Interpreter','latex','FontSize',20);
lg3 = legend('$m_{t+1} = 1$','$m_{t+1} = 2$');
lg3.Interpreter = 'latex';
lg3.FontSize = 12;
xlabel('$\mathbf{x_{t}}$','Interpreter','latex','FontSize',16)
ylabel('$\mathbf{P(m_{t+1}|x_{t},m_{t})}$','Interpreter','latex','FontSize',16)

figure()
H2 = [p_m(1).data(1,:)' p_m(1).data(2,:)' p_m(2).data(1,:)' p_m(2).data(2,:)'];
subplot(2,1,1);
bar(test_pt_x,H2(:,1:2));
title('P(m_{t+1}|x_{t},m_{t})','FontSize',20);
legend('P(m_{t+1} = 1|m_{t} = 1)','P(m_{t+1} = 2|m_{t} = 1)')
xlabel('x_{t}','FontSize',16)
ylabel('P(m_{t+1})','FontSize',16)
subplot(2,1,2);
bar(test_pt_x,H2(:,3:4));
title('P(m_{t+1}|x_{t},m_{t})','FontSize',20);
legend('P(m_{t+1} = 1|m_{t} = 2)','P(m_{t+1} = 2|m_{t} = 2)')
xlabel('$\mathbf{x_{t}}$','Interpreter','latex','FontSize',12)
ylabel('P(m_{t+1})')

%% step 7 - test the particle filter to see mode predictions

% create a mode map
if rep_density == 0
    MAXX = 20; % interval size
    x = linspace(0,10,MAXX);
elseif rep_density ==1
    x = start_x;
    MAXX = length(x);
    H = [];
end

N = 50;%number of samples for particle filtering
Q = 50;% number of samples for importance sampling
mag = 1;% for importance sampling in state space

for c = 1:2 % prior mode - over all m(t)
    ModeOutput = zeros(MAXX,3);% [mode output, P(1), P(2)]
    for i = 2:MAXX % over all x(t)
        obs = x(i);% y(t)
        feature = x(i-1);% f(t-1)
        
        % generate N points randomly
        f_next = [feature+mag*randn(N,1)];% guessed distribution of x(t) evaluated at t-1 by distributing around x(t-1)
        particles_x = f_next; % initialization only - refers to the particles after correction only - estimate the true state
        % correction using y(t) and propagating to get x(t) and m(t)
        particles_m = map_mode1(obs,c);
        
        ModeOutput(i,2) = sum(particles_m < 2)/N; % probability of mode 1
        ModeOutput(i,3) = sum(particles_m > 1)/N; % probability of mode 2
        consensus = mode(particles_m);
        ModeOutput(i,1) = consensus;
        
        if rep_density == 1
            % for getting representative mode density for 4 points
%             subplot(2,2,i);
%             histogram([particles_m;1;2],'Normalization','probability');
%             title(['x_{t}: ',num2str(x(i)),' | m_{t}: ' num2str(c)]); 
%             ylabel('P(m_{t+1}|x_{t},m_{t})');
%             xlabel('m_{t+1}');
            H = [H particles_m];
        end
    end
    ModeData(c).ModeOutput = ModeOutput;
end

if rep_density ==1 
    H1 = zeros(size(H,2)/2,4);
    for a = 1:size(H,2)/2
        H1(a,:) = [sum(H(:,a)<2)/size(H,1) sum(H(:,a)>1)/size(H,1) sum(H(:,a+size(H,2)/2)<2)/size(H,1) sum(H(:,a+size(H,2)/2)>1)/size(H,1)]; % [P(1,1) P(2,1) P(1,2) P(2,2)]
    end
    figure();
    subplot(2,1,1);
    bar(H1(:,1:2));
    title('P(m_{t+1}|x_{t},m_{t})');
    legend('P(m_{t+1} = 1|m_{t} = 1)','P(m_{t+1} = 2|m_{t} = 1)','FontSize',20)
    xlabel('x_{t}','FontSize',16)
    ylabel('P(m_{t+1})','FontSize',16)
    
    subplot(2,1,2);
    %figure();
    bar(H1(:,3:4));
    title('P(m_{t+1}|x_{t},m_{t})','FontSize',20);
    legend('P(m_{t+1} = 1|m_{t} = 2)','P(m_{t+1} = 2|m_{t} = 2)')
    xlabel('x_{t}','FontSize',16)
    ylabel('P(m_{t+1})','FontSize',16)
    
end

if rep_density == 0
    % for getting mode density over state space
    figure()
    plot(x,ModeData(1).ModeOutput(:,1))
    hold on;
    plot(x,ModeData(2).ModeOutput(:,1))
    title('Mode chosen given the prior mode and x')
    legend('Prior mode = 1','Prior mode = 2')

    % probability of next mode 1
    figure()
    plot(x,ModeData(1).ModeOutput(:,2)); % given prior mode 1
    hold on;
    % plot(x,ModeData(2).ModeOutput(:,2)); % given prior mode 2
    % hold on;
    % % probability of next mode 2
    % plot(x,ModeData(1).ModeOutput(:,3)); % given prior mode 1
    % hold on;
    plot(x,ModeData(2).ModeOutput(:,3)); % given prior mode 2
    title('P(m_{t+1}|x_{t},m_{t})');
    legend('P(1|1)','P(2|2)');
end
%% Step 8 - Propagating both state and mode using the particle filter

% initial position and mode (at t=0)
x_0 = 0;
m_0 = 1;
% initial readings at (t=1)
y_t = x_0;
% x_t = x_0;
path = [x_0 m_0];
feature = x_0;
f_next = [feature+mag*randn(N,1)];% representation of propagated/predicted state at t+1
particles_x = f_next;% representation of corrected state at t=0
particles_m = ones(N,1);
for n = 1:N
    particle_propogation(n).state = particles_x(n,1);% the state stores a 1X1 current position
    particle_propogation(n).particles_m = particles_m(n);
end
% particle_propogation = [particles_x'];
Pm_t = [];
for stp = 1:100
    y_t = y_t + v(m_0)*T+ noise_mag*randn();% measurement of the next state
%     x_t = x_t + v(m_0)*T;% true next state
    particles_m = map_mode1(y_t,m_0);
    consensus = mode(particles_m);
    m_0 = consensus;
    path = [path;[y_t m_0]];
%     actual_mode = normpdf(x_t,2,1) + normpdf(x_t,6,1);
%     actual_mode = (actual_mode > 0.3) + 1;
    % fill the Pm_t - number of particles in each mode at time t
    Pm_t_cur = [];
    for o = 1:length(classes)
        Pm_t_cur = [Pm_t_cur;sum(particles_m  == o)];
    end
    Pm_t_cur = Pm_t_cur/sum(Pm_t_cur);
    Pm_t = [Pm_t [Pm_t_cur;m_0]];
    
    % particle propogation happening inside the map_mode1 function
    % as part of the resampling
    for n = 1:N
        particle_propogation(n).state = [particle_propogation(n).state; particles_x(n,:)];
        particle_propogation(n).particles_m = [particle_propogation(n).particles_m; particles_m(n)];
    end
    disp(stp);
end
%%
figure();
pl(1) = imagesc(Pm_t(1:end-1,:));
hold on;
pl(2) = plot(Pm_t(end,:),'r','LineWidth',5);set(gca,'FontSize',12);set(gca,'YTickLabel',[' ';'1';' ';'2'])
[~,ii] = max(Pm_t(1:end-1,:));
correct_class = sum(ii==Pm_t(end,:));
err_class = sum(ii==Pm_t(end,:))/length(Pm_t);
title('Mode prediction and Actual mode','FontSize',20);
xlabel('$\mathbf{t}$','Interpreter','latex','FontSize',16);
ylabel('$\mathbf{m_t}$','Interpreter','latex','FontSize',16);
lg4 = legend('Actual mode');
lg4.FontSize = 12;
%%
% plot the propagation
figure();
sz1 = T*[1:size(particle_propogation(1).state,1)];
for iter = 1:length(particles_x)
    g(1) = scatter(particle_propogation(iter).state,sz1,50,'r','filled','DisplayName','Particles: x^{i}_t)');
    hold on;
end
% actual motion implemented
g(2) = scatter(path(:,1),sz1,50,'k','DisplayName','Robot: x_t');
legend(g(1:2));
title('Particles and robot motion','FontSize',20)
ylabel('time (s)','FontSize',16);
axis equal

% plot the propagation
% figure();
% sz1 = T*[1:size(particle_propogation(:,1),1)];
% for iter = 1:length(particles_x)
%     h(4) = plot(particle_propogation(:,iter),sz1,'DisplayName','x^{i}_t)');
%     hold on;
% end
% % actual motion implemented
% h(5) = scatter(path(:,1),sz1,'k','filled','DisplayName','x_t');
% title('Particles and actual motion','FontSize',20)
% ylabel('time (s)','FontSize',16);
% xlabel('x_{t}','FontSize',16);
% 
figure();
scatter(path(:,1),path(:,2))
