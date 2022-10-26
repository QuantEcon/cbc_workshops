%% Inflation targeting under OBC
%  Fernández and Rondón (2021)
%  In this file, we use VFI to obtain the policy function to solve the OER
%  regime.
%  This file replicates SGU (2016): ``Downward Nominal Wage Rigidity,
%  Currency Pegs, And Involuntary Unemployment,'' (2016), Journal of Political Economy.
% 

clear all
clc

%% Load Parameters and transition matrix

load Parameters.mat sigma hbar a epsilon alpha beta Dmin Dmax Dn rstar Yn Rn NumSim burn
format longg
eval(['filename = ''TransitionMatrix_' num2str(beta, '%5.10f') '.mat'''  ])
eval(['load ' filename '  Tran S'  ])


%% Forming grids for exogenous states yT, rT and dT

rgrid = exp(S(:,2))*(1+rstar)-1; %interest rate in level 
ygrid = exp(S(:,1)); %level of tradable output
ny    = numel(ygrid);
dgrid = linspace(Dmin,Dmax,Dn)';

%% Computing every option for consumption of tradables


n = ny*Dn; % Every posible realization of YxR (YnXRn) times total points in D grid: A (YnxRnxDn) vector
d = repmat(dgrid',ny,1);
d = d(:);
r = repmat(rgrid,Dn,1);
y = repmat(ygrid,Dn,1);

% Construct cTtry = all possibilties for yT-d+dp/(1+r)
cTtry = bsxfun(@ldivide,(1+r),dgrid'); %this is just the part dp/(1+r)
cTtry = bsxfun(@plus,cTtry,y-d);  %! A (YnxRnxDn)xDn Vector...a policy option for each option of debt in tomorrow's debt

if min(max(cTtry,[],2))<0
    error('Natural debt limit violated')
end

% Total consumption:

ctry = (a * cTtry.^(1-1/epsilon) + (1-a) * (hbar^alpha).^(1-1/epsilon)).^(1/(1-1/epsilon));%composite consumption

% Utility Function:

utry = (ctry.^(1-sigma) - 1)./(1-sigma);

utry(cTtry<=0) = -inf; % Make sure we will never choose a point with negative consumption

clear ctry cTtry
%% Value function iteration

tol1 = 1e-8;     % Tolerance 1 for main loop
tol2 = 1e-8;     % Tolerance 2 for inside loop

v = zeros(ny,Dn);    % Value Function
vnew = v;
dpix = zeros(ny,Dn); % D_{t+1} policy function
dpixnew = dpix;     

delta = 1;
iter  = 1;
iter2  = 1;

while delta > tol1
    
    [vnew(:), dpixnew(:)] = max(utry + beta*repmat(Tran *  v,Dn,1),[],2);
    
    delta = max(abs(vnew(:)-v(:))) + max(abs(dpixnew(:)-dpix(:)));
    
    v = vnew;
    % Accelerate convergence if dpix has converged
    if (dpix==dpixnew) & (delta>tol2) 
        delta2 = 1;
        picker = sub2ind([n Dn],(1:n)',dpix(:));    
        u = utry(picker);
        v1 = zeros(ny,Dn);
        while delta2 > tol2 
            Vtemp2 = repmat(Tran * v,Dn,1);
            v1(:) = u + beta * Vtemp2(picker);
            clear Vtemp2
            delta2 = max(abs(v(:)-v1(:)));
            v = v1;
            if mod(iter2,100)==0
                 disp([iter2 delta2])
            end
            iter2 = iter2 + 1;
        end 
    end 
    
    dpix = dpixnew;
    if mod(iter,100)==0
        disp([iter delta])
    end
    iter = iter+1;
end

dp = dgrid(dpix);

disp("done");

%% Save Results

format longg
%eval(['filename = ''/repositorio/crondon/ITunderOBC/vfi_oer_' num2str(beta, '%5.10f') '.mat'''  ])
%eval(['save ' filename ' v dpix dp'])

