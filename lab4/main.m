clear all; close all; clc

rng('default');
tstart = tic;
Kquant = 8;  % Number of quantization levels
Nstates = 8;  % Number of states in the HMM
ktrain = [1,2,3,4,5,6,7];  % Indexes of patients for training
ktest = [8,9,10];  % Indexes of patients for testing
[hq, pq] = pre_process_data(Nstates, Kquant, ktrain);  % Generate the...
                                                       % quantized signals
telapsed = toc(tstart);
disp(['first part, elapsed time ', num2str(telapsed), ' s'])
%% HMM training phase....

plot([hq{1}])

%% HMM testing phase....
