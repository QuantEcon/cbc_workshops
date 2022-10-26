 %% Inflation targeting under OBC
%  Fernández and Rondón (2021)
%  In this file, we set the necessary parameters to find the dynamic solution to each of the three policy enviroments proposed in the
%  paper.

%% Set parameters

clear all
clc

% Structural Parameters

sigma = 5;      % Inverse of intertemporal elasticity of consumption %! SGU(2016) uses sigma = 5;
hbar  = 1;      % Labor Endowment
a     = 0.19;   % Share of tradables %! It comes from Argentina's calibration in SGU (2016)
epsilon = 0.43;    % Elasticity of substition between tradables and nontradables %! SGU (2016) uses 0.44
alpha = 0.75;   % Labor share in nontraded sector 

% Calibrated Parameters

beta  = 0.95710000;     % Quarterly subjective discount factor %! Set to match D/Y = 0.364
gamma = 0.96;       % Quarterly Parameter governing the wage downward rigidity 
targetPi = 0.0;   % Set inflation target %? Should it be annualized?
Pi    = 1 + targetPi/4;  % Quarterly Inflation Rate    
Growth = 1+0.020758/4 ;
%Gamma0 = 0.99;
%gamma = Gamma0./(Pi);

%% State-Space Parameters
%! Initially set to match SGU calibration
Dmin = -5.0;      % Minimum Value Debt Grid
Dmax =  5.4;     % Maximum Value Debt Grid (Natural debt limit) %! computed from tpm
Dn   =  501;      % Debt Grid Points


%* For the currency peg, a second endogenous state emerges, past real wage.
%! Initially set to match SGU calibration
Wmin = 0.1;     % Min Value Wage Grid
Wmax = 14.3;     % Max Value Wage Grid
WminSIT = 0.3;
WmaxSIT = 0.8;
Wn   = 500;     % Wage Grid Points

%% Paramenters for simulations

start = 5;   % #s of periods in episodes window
nstd  = 2;   % Number of std. deviations considered for shock

%% Stochastic Processes Parameters

load VARparam.mat bLAC5 sLAC5 rss_LAC5 bPP sPP rPP bLAC5Y sLAC5Y rss_LAC5Y
Yn     = 21;       % Grid size for tradable output
Rn     = 11;       % Grid size for foreign interest rate

LAC    = 2;

if LAC == 0
    A  = bLAC5;
    Sigma  = chol(sLAC5); 
    rstar = rss_LAC5/400; % Average real interest rate
elseif LAC == 1
    A      = bPP;
    Sigma  = sPP; % has to be upper triangular
    rstar  = rPP/100;
else 
    A      = bLAC5Y;
    Sigma  = chol(sLAC5Y); % has to be upper triangular
    rstar  = rss_LAC5Y/100; % Average real interest rate;
end

Mu     = [0; 0];
NumSim = 10e6;
burn   = 1e6;

%clear bPP sPP rPP
save('Parameters','-v7.3') ;

