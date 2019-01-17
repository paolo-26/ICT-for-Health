clear; close all; clc
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

%% HMM training phase

TOL = 1e-3;  % Tolerance
MAX_ITER = 200;  % Maximum number of iterations
ALG = 'BaumWelch';

transGuess = rand(nStates,nStates);
% Normalize rows
s = sum(transGuess, 2);
for iRow = 1:nStates
    transGuess(iRow,:) = transGuess(iRow,:) / s(iRow);
end

emisGuess = rand(nStates,nStates);
% Normalize rows
s = sum(emisGuess, 2);
for iRow = 1:nStates
    emisGuess(iRow,:) = emisGuess(iRow,:) / s(iRow);
end
%emisGuess = 1/8*ones(8,8);

% Health machine training
[transH, emisH] = hmmtrain(hq(kTrain), transGuess, emisGuess,...
                           'algorithm', ALG,...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);

% Parkinson machine training
[transP, emisP] = hmmtrain(pq(kTrain), transGuess, emisGuess,...
                           'algorithm', ALG,...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);
figure(1)
heatmap(transH); colorbar
%axis square;
title({'Transition matrix', 'Healthy patients'})
xlabel('state'); ylabel('state')

% figure(2)
% pcolor(emisH); colorbar
% axis square; title({'Emission matrix (training)', 'Healthy patients'})
% xlabel('state'); ylabel('state')

figure(2)
heatmap(transP); colorbar
%axis square;
title({'Transition matrix', 'Ill patients'})
xlabel('state'); ylabel('state')

% figure(4)
% pcolor(emisP); colorbar
% axis square; title({'Emission matrix (training)', 'Ill patients'})
% xlabel('state'); ylabel('state')

%% HMM testing phase

specificityTrain = 0;
for i = kTrain
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        specificityTrain = specificityTrain + 1/length(kTrain);
    end
end

sensitivityTrain = 0;
for i = kTrain
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        sensitivityTrain = sensitivityTrain + 1/length(kTrain);
    end
end

specificityTest = 0;
for i = kTest
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        specificityTest = specificityTest + 1/length(kTest);
    end
end

sensitivityTest = 0;
for i = kTest
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        sensitivityTest = sensitivityTest + 1/length(kTest);
    end
end

clear transGuess

p = 0.9;
q = (1 - p) / (nStates - 1);
qs = q*ones(1, nStates-1);
v = [p qs];
transGuess = zeros(nStates,nStates);
for i = 1:nStates
    
   transGuess(i,:) =  circshift(v,1);
   v = transGuess(i,:);
end

% Health machine training
[transH, emisH] = hmmtrain(hq(kTrain), transGuess, emisGuess,...
                           'algorithm', ALG,...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);

% Parkinson machine training
[transP, emisP] = hmmtrain(pq(kTrain), transGuess, emisGuess,...
                           'algorithm', ALG,...
                           'tolerance', TOL,...
                           'maxiterations', MAX_ITER);
figure(3)
heatmap(transH); colorbar
%axis square;
title({'Transition matrix', 'Healthy patients'})
xlabel('state'); ylabel('state')

% figure(2)
% pcolor(emisH); colorbar
% axis square; title({'Emission matrix (training)', 'Healthy patients'})
% xlabel('state'); ylabel('state')

figure(4)
heatmap(transP); colorbar
%axis square;
title({'Transition matrix', 'Ill patients'})
xlabel('state'); ylabel('state')

% figure(4)
% pcolor(emisP); colorbar
% axis square; title({'Emission matrix (training)', 'Ill patients'})
% xlabel('state'); ylabel('state')

specificityTrainPQ = 0;
for i = kTrain
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        specificityTrainPQ = specificityTrainPQ + 1/length(kTrain);
    end
end

sensitivityTrainPQ = 0;
for i = kTrain
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        sensitivityTrainPQ = sensitivityTrainPQ + 1/length(kTrain);
    end
end

specificityTestPQ = 0;
for i = kTest
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        specificityTestPQ = specificityTestPQ + 1/length(kTest);
    end
end

sensitivityTestPQ = 0;
for i = kTest
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        sensitivityTestPQ = sensitivityTestPQ + 1/length(kTest);
    end
end