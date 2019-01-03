clear all
close all
%%
rng('default');
tstart=tic;
Kquant=8;% number of quantization levels
Nstates=8;% number of states in the HMM
ktrain=[1,2,3,4,5,6,7];% indexes of patients for training
ktest=[8,9,10];% indexes of patients for testing
[hq,pq]=pre_process_data(Nstates,Kquant,ktrain);% generate the quantized signals
telapsed = toc(tstart);
disp(['first part, elapsed time ',num2str(telapsed),' s'])
%% HMM training phase....
%% HMM testing phase....

