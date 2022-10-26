function [Pi,S,Xvec,UB] = tpm1(A,omega,N,T,Tburn, UB,trim);
%[Pi,S] = tpm(A,omega,N,T,Tburn, UB,trim)
% discretizes the AR(1) process:
%x_t = A x_t-1  + omega e_t,    (1)
%where x_t is m-by-1, A is m-by-m,  omega is m-by-r, and e is an r-by-1
%white noise vector with mean zero and identity variance-covariance matrix.
%Pi is the n-by-n transition probability matrix of the discretized state.
%S is an n-by-m matrix. Element (i,j) of S is the discretized value of the j-th element of x_t in state i. 
%N is an m-by-1 vector. Element i of N indicates the number of grid points in the discretization of element i of x_t. 
%n = N(1) * N(2) * ... * N(m); 
%The default value of N is all elements equal to 10. 
%All grids contain equally spaced points and 
%are symmetric around 0 
%with upper bound UB and lower bound -UB. 
%T is the length of the simulated time series from (1) used to calculate Pi. The default value of T is 1 million.
%Tburn is the number of burn-in draws from simulated time series. Default 0.1 million.
%UB is an m-by-1 vector. Element (i) of UB contains the upper bound of the
%grid for the i-th element of the vector x_t. The default value is sqrt(10)*std(x_t(i)). 
%trim =1 trims Pi and S by removing states that are never visited. Any other value of trim results in no trimming.   Default is trim=0.Trimming 
%removes column i and row i of Pi as well as row i of S if all elements of column i of Pi  are zero. 
%To use default values set the input argument to empty, for example, [Pi,S] = tpm(A,omega,[],[],[],[],1);
%For a derivation of the matrices Pi and S, see ``Finite-State Approximation Of  VAR Processes:  A Simulation Approach'' by Stephanie Schmitt-Grohé and Martín Uribe, July 11, 2010. 
%calls: subfunction mom
%(c) Stephanie Schmitt-Grohé and Martín Uribe, July 11, 2010. 

if nargin<7|isempty(trim)
    trim = 0; %no trimming
end

if nargin<6|isempty(UB)
    Sigg=mom(eye(size(A)),A,omega'*omega) %variance matrix of AR process
    sigg = sqrt(diag(Sigg)) %Unconditional  standard deviation  of AR process
    UB = sqrt(10)*sigg%upper bound for grid. 
end

if nargin<5|isempty(Tburn)
    Tburn = 1e5
end

if nargin<4|isempty(T)
    T = 1e+6
end

if nargin<3|isempty(N)
    N = 10*ones(size(A,1),1);
end

m = size(A,1);
r = size(omega,2);

%Grid 
for j=1:m
    V(j) ={ -UB(j): 2*UB(j) / (N(j)-1) : UB(j)};
end

n = prod(N); %total number of possible values of the discretized state

S = zeros(n,m); 
for i=1:m
    if i==1
    ans = V{i};
    else
    repmat(V{i},[prod(N(1:i-1)) 1]);
    end
    ans(:);
    if i<m
        repmat(ans,[prod(N(i+1:end)) 1]);
    end
    S(:,i) = ans;
end

Pi = zeros(n);

%initialize the  state
x0 = zeros(m,1); %initialize simulated time series
xx = repmat(x0',n,1);
d = sum((S-xx).^2,2);
[~,i] = min(d);

%randn('state',0)
Xvec = zeros(m,T+Tburn) ;
for t=1:T+Tburn
    x = A*x0 + omega'*randn(r,1);
    Xvec(:,t) = x ;
    %Normality is not required. The command randn can be changed to any other random number generator with mean 0 and unit standard deviaiton. 
    xx = repmat(x',n,1);
    d = sum((S-xx).^2,2);
    [~,j] = min(d);
    if t>Tburn
    Pi(i,j) = Pi(i,j)+1;
    end
    x0 = x;
    i = j;
end

if trim==1
    z = find(sum(Pi)>0);
    Pi = Pi(z,:);
    Pi = Pi(:,z);
    S = S(z,:);
end

z = find(sum(Pi,2)==0);
Pi(z,:) = 1;

n1 = size(Pi,1);
for i=1:n1
    Pi(i,:) = Pi(i,:)./sum(Pi(i,:));
end

Pi(z, :) =0;

function [sigyJ,sigxJ]=mom(gx,hx,varshock,J, method)
%[sigyJ,sigxJ]=mom(gx,hx,varshock,J, method)
% Computes the unconditional variance-covariance matrix of x(t) with x(t+J), that is sigxJ=E[x(t)*x(t+J)'], 
%and the unconditional variance covariaance matrix of y(t) with y(t+J), that is sigyJ=E[y(t)*y(t+J)']
% where x(t) evolves as
% x(t+1) = hx x(t) + e(t+1)
%and y(t) evolves according to 
% y(t) = gx x(t)
%where Ee(t)e(t)'=varshock
%The parameter J can be any integer
%method =1 : use doubling algorithm
%method neq 1 : use algebraic method
%(c) Stephanie Schmitt-Grohe and Martin Uribe, April 18, 1990, renewed January 24, 2000 and August 18, 2001. 


if nargin<4
    J=0;
end


if nargin<5
    method =1;
end


if method == 1 
    %disp('method=doubling')

    %Doubling algorithm
    hx_old=hx;
    sig_old=varshock;
    sigx_old=eye(size(hx));
    diferenz=.1;
    while diferenz>1e-25;
        sigx=hx_old*sigx_old*hx_old'+sig_old;
        diferenz = max(max(abs(sigx-sigx_old)));
        sig_old=hx_old*sig_old*hx_old'+sig_old;
        hx_old=hx_old*hx_old;
        sigx_old=sigx;
    end    %while diferenz
else
%Algebraic method
%Get the variance of x
    disp('method=kronecker')
    sigx = zeros(size(hx));
    F = kron(hx,hx);
    sigx(:) = (eye(size(F))-F)\varshock(:);
end   %if method
%Get E{x(t)*x(t+J)'}
sigxJ=hx^(-min(0,J))*sigx*(hx')^(max(0,J));

%Get E{y(t)*y(t+J)'}
sigyJ=real(gx*sigxJ*gx');