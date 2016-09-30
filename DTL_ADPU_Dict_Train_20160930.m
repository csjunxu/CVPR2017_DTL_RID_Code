clear;clc;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');

load Data/EMGM_8x8_100_knnNI2BS500Train_20160722T082406.mat;
% Parameters Setting
par.rho = 0.05;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nup              =      0;
par.epsilon         =        5e-3;
par.cls_num            =    cls_num;
par.step               =    2;
par.win                =    8;
par.nIter           =       100;
par.t0              =       5;
par.K               =       256;
par.L               =       par.win * par.win;
param.K = par.K;
param.iter=300;
param.lambda = par.lambda1;
param.lambda2 = par.lambda2;
param.L = par.win * par.win;
flag_initial_done = 0;
paramsname = sprintf('Data/params_%s.mat',datestr(now, 30));
save(paramsname,'par','param');

% Initiate Dictionary
Dini = [];
for i = 1 : par.cls_num
    XN_t = double(Xn{i});
    XC_t = double(Xc{i});
    XN_t = XN_t - repmat(mean(XN_t), [par.win^2 1]);
    XC_t = XC_t - repmat(mean(XC_t), [par.win^2 1]);
    fprintf('Dictionary and Transformation Learning: Cluster: %d\n', i);
    Dc = mexTrainDL(XC_t, param);
    Dini{i} = Dc;
    Dict_BID_Initial = sprintf('Data/Dict_DcTL_Initial_%s.mat', datestr(now, 30));
    save(Dict_BID_Initial,'Dini');
    Dc = Dini{i};
    clear Dini;
    Wn = eye(size(Dc, 2));
    Wc = eye(size(Dc, 2));
    Alphan = mexLasso(XN_t, Dc, param);
    Alphac = mexLasso(XC_t, Dc, param);
    fprintf('Dictionary and Transformation Learning: Cluster: %d\n', i);
    [Alphac, Alphan, XC_t, XN_t, Dc, Wc, Wn, Uc, Un, Vn, f] = DcTL_ADPU_20160930(Alphac, Alphan, XC_t, XN_t, Dc, Wc, Wn, par);
    Dict.DC{i} = Dc;
    Dict.WC{i} = Wc;
    Dict.WN{i} = Wn;
    Dict.UC{i} = Uc;
    Dict.UN{i} = Un;
    Dict.VN{i} = Vn;
    Dict.f{i} = f;
    Dict_BID_backup = sprintf('Data/DTL_RID_Dict_backup_nup0_%s.mat',datestr(now, 30));
    save(Dict_BID_backup,'Dict');
end