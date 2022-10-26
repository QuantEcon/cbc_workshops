%% Inflation targeting under OBC
%  Fernández and Rondón (2021)
%  In this file, we discretize the AR(1) processes por ln(Yt^T) and ln()(1+Rt)/(1+r)) using tpm.m. 
%
clear all
clc

rng(5)

load 'Parameters.mat' Yn Rn A Sigma Mu NumSim burn rstar beta;
[Tran,S,Xvec,UB] = tpm1(A, Sigma, [Yn;Rn], NumSim/10*2, burn/10,[],1); % Fundamental Transition Matrix, Matrix of states and Simulation

%[Tran,S,Xvec,UB] = tpm(A, Sigma, [Yn;Rn], NumSim, burn); % Fundamental Transition Matrix, Matrix of states and Simulation
% Natural debt limit: Level of external debt that can be supported with
% zero tradable consumption when the household perpetually receives the
% lowest possible realization of tradable endowment
% dbar = yTmin*(1+rmax)/rmax

rgrid = exp(unique(S(:,2)))*(1+rstar)-1; %interest rate in level 
ygrid = exp(unique(S(:,1)));             %level of tradable output

dbar = min(ygrid)*(1 + max(rgrid))/max(rgrid)

clearvars -except Tran S Xvec UB beta 

format longg
eval(['filename = ''TransitionMatrix_' num2str(beta, '%5.10f') '.mat'''  ])
eval(['save ' filename ])

