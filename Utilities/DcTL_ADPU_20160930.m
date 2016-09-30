% Main Function of Coupled Dictionary Learning
% Input:
% Alphap,Alphas: Initial sparse coefficient of two domains
% Xp    ,Xs    : Image Data Pairs of two domains
% Dp    ,Ds    : Initial Dictionaries
% Wp    ,Ws    : Initial Projection Matrix
% par          : Parameters
%
% Output
% Alphap,Alphas: Output sparse coefficient of two domains
% D    : Output Dictionaries
% Up    ,Us    : Output Projection Matrix for Alpha
%

function [Alphap, Alphas, Xp, Xs, Dc, Wp, Ws, Up, Us, Vs, f] = DcTL_ADPU_20160930(Alphap, Alphas, Xp, Xs, Dc, Wp, Ws, par)

%% parameter setting
param.lambda        = 	    par.lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       par.lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
f = 0;

%% Initialize Us, Up as I

Us = Ws;
Up = Wp;

%% Initialize Ps as 0 matrix
Vs = zeros(size(Xs));

%% Iteratively solve D A U

for t = 1 : par.nIter
    
    %% Updating Alphas and Alphap
    f_prev = f;
    Alphas = mexLasso([Xs - Vs; par.sqrtmu * Up * full(Alphap)], [Dc; par.sqrtmu * Us],param);
    Alphap = mexLasso([Xp; par.sqrtmu * Us * full(Alphas)], [Dc; par.sqrtmu * Up],param);
    dictSize = par.K;
    
    %% Updating Dc
    for i=1:dictSize
        ai        =    Alphap(i,:);
        Y         =   Xp - Dc * Alphap + Dc(:,i) * ai;
        di        =    Y * ai';
        di        =    di ./ (norm(di,2) + eps);
        Dc(:,i)   =    di;
    end
    
    %% Updating Ps
    Vs = Xs - Dc * Alphas;
    
    %% Updating Ws and Wp => Updating Us and Up
    Us = (1 - par.rho) * Us  + par.rho * Up * Alphap * Alphas' / ( Alphas * Alphas' + par.nu * eye(size(Alphas, 1)));
    Up = (1 - par.rho) * Up  + par.rho * Us * Alphas * Alphap' / ( Alphap * Alphap' + par.nu * eye(size(Alphap, 1)));
    Ws = Up /Us;
    Wp = Us /Up;
    
    %% Find if converge (NEED MODIFICATION)
    P1 = Xp - Dc * Alphap;
    P1 = P1(:)'*P1(:) / 2;
    P2 = par.lambda1 *  norm(Alphap, 1);
    P3 = Us * Alphas - Up * Alphap;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Up, 'fro');
    fp = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
    
    P1 = Xs - Dc * Alphas - Vs;
    P1 = P1(:)'*P1(:) / 2;
    P2 = par.lambda1 *  norm(Alphas, 1);
    P3 = Us * Alphas - Up * Alphap;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Us, 'fro');
    P5 = par.nup * norm(Vs, 'fro');
    fs = 1 / 2 * P1 + P2 + par.mu * (P3 + P4 + P5);
    
    f = fp + fs;
    
    %% if converge then break
    if (abs(f_prev - f) / f < par.epsilon)
        break;
    end
    fprintf('Energy: %d\n',f);
    save tempDcTL_RID_Dict_ADPU.mat Dc Us Up Ws Wp par param;
end

