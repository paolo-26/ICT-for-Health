clear all; close all; clc
format short;

rng('default');
tStart = tic;
nQuant = 8;  % Number of quantization levels
nStates = 8;  % Number of states in the HMM
kTrain = [1, 2, 3, 4, 5, 6, 7];  % Indexes of patients for training
kTest = [8, 9, 10];  % Indexes of patients for testing
[hq, pq] = pre_process_data(nStates, nQuant, kTrain);  % Generate the...
                                                       % quantized signals
tElapsed = toc(tStart);
disp(['first part, elapsed time ', num2str(tElapsed), ' s'])

%% HMM training phase....
TOL = 1e-3;  % Tolerance
MAX_ITER = 200;  % Maximum number of iterations

transGuess = rand(8,8);
% Normalize rows
s = sum(transGuess, 2);
for iRow = 1:8
    transGuess(iRow,:) = transGuess(iRow,:) / s(iRow);
end

emisGuess = rand(8,8);
% Normalize rows
s = sum(emisGuess, 2);
for iRow = 1:8
    emisGuess(iRow,:) = emisGuess(iRow,:) / s(iRow);
end

% Health machine training
[transH, emisH] = hmmtrain(hq(kTrain), transGuess, emisGuess,...
                           'algorithm', 'baumwelch',...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);

% Parkinson machine training
[transP, emisP] = hmmtrain(pq(kTrain), transGuess, emisGuess,...
                           'algorithm', 'baumwelch',...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);
figure(1)
pcolor(transH); colorbar
axis square; title({'Transition matrix (training)', 'Healthy patients'})
xlabel('state'); ylabel('state')

figure(2)
pcolor(emisH); colorbar
axis square; title({'Emission matrix (training)', 'Healthy patients'})
xlabel('state'); ylabel('state')

figure(3)
pcolor(transP); colorbar
axis square; title({'Transition matrix (training)', 'Ill patients'})
xlabel('state'); ylabel('state')

figure(4)
pcolor(emisP); colorbar
axis square; title({'Emission matrix (training)', 'Ill patients'})
xlabel('state'); ylabel('state')

%% HMM testing phase....

% PSTATES = hmmdecode(hq{8}, trans, emis);

