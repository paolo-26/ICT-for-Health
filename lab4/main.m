clear; close all; clc
format short;

rng('default');
tStart = tic;

comb = 1;  % Choose from [1 2 3]
nQuant = 8;  % Number of quantization levels
nStates = 8;  % Number of states in the HMM

% Indexes of patients for training.
kTrainT = [1 2 3 4 5 6 7; 4 5 6 7 8 9 10; 1 2 3 4 8 9 10];

% Indexes of patients for testing.
kTestT = [8 9 10; 1 2 3; 5 6 7];

% Select set depennding on combination.
kTrain = kTrainT(comb,:);
kTest = kTestT(comb,:);

% Generate the quantized signals.
[hq, pq] = pre_process_data(nStates, nQuant, kTrain);

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

emisGuess = rand(nStates,nQuant);
% Normalize rows
s = sum(emisGuess, 2);
for iRow = 1:nStates
    emisGuess(iRow,:) = emisGuess(iRow,:) / s(iRow);
end
%emisGuess = 1/8*ones(8,8);

tempo1 = tic;
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
% figure(1)
% heatmap(transH); colorbar
% %axis square;
% title({'Transition matrix - Healthy',...
%       ['nQuant = ' , num2str(nQuant),...
%       '; nStates = ' , num2str(nStates)]})
% xlabel('state'); ylabel('state')
%
% figure(2)
% pcolor(emisH); colorbar
% axis square; title({'Emission matrix (training)', 'Healthy patients'})
% xlabel('state'); ylabel('state')
% 
% figure(2)
% heatmap(transP); colorbar
% %axis square;
% title({'Transition matrix - Parikinson''s disease',...
%       ['nQuant = ' , num2str(nQuant),...
%       '; nStates = ' , num2str(nStates)]})
% xlabel('state'); ylabel('state')
%
% figure(4)
% pcolor(emisP); colorbar
% axis square; title({'Emission matrix (training)', 'Ill patients'})
% xlabel('state'); ylabel('state')

%% HMM testing phase

trainSpec = 0;
for i = kTrain
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        trainSpec = trainSpec + 1/length(kTrain);
    end
end

trainSens = 0;
for i = kTrain
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        trainSens = trainSens + 1/length(kTrain);
    end
end

testSpec = 0;
for i = kTest
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        testSpec = testSpec + 1/length(kTest);
    end
end

testSens = 0;
for i = kTest
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        testSens = testSens + 1/length(kTest);
    end
end

tempo2 = toc(tempo1);

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

tempo3 = tic;
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
                       
% figure(3)
% heatmap(transH); colorbar
% %axis square;
% title({'Transition matrix - Healthy',...
%       ['nQuant = ' , num2str(nQuant),...
%       '; nStates = ' , num2str(nStates)]})
% xlabel('state'); ylabel('state')
%
% figure(2)
% pcolor(emisH); colorbar
% axis square; title({'Emission matrix (training)', 'Healthy patients'})
% xlabel('state'); ylabel('state')
%
% figure(4)
% heatmap(transP); colorbar
% %axis square;
% title({'Transition matrix - Parikinson''s disease',...
%       ['nQuant = ' , num2str(nQuant),...
%       '; nStates = ' , num2str(nStates)]})
% xlabel('state'); ylabel('state')
%
% figure(4)
% pcolor(emisP); colorbar
% axis square; title({'Emission matrix (training)', 'Ill patients'})
% xlabel('state'); ylabel('state')

PQtrainSpec = 0;
for i = kTrain
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        PQtrainSpec = PQtrainSpec + 1/length(kTrain);
    end
end

PQtrainSens = 0;
for i = kTrain
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        PQtrainSens = PQtrainSens + 1/length(kTrain);
    end
end

PQtestSpec = 0;
for i = kTest
    [~, logpH] = hmmdecode(hq{i}, transH, emisH);
    [~, logpP] = hmmdecode(hq{i}, transP, emisP);
    if logpH > logpP
        PQtestSpec = PQtestSpec + 1/length(kTest);
    end
end

PQtestSens = 0;
for i = kTest
    [~, logpH] = hmmdecode(pq{i}, transH, emisH);
    [~, logpP] = hmmdecode(pq{i}, transP, emisP);
    if logpH < logpP
        PQtestSens = PQtestSens + 1/length(kTest);
    end
end

tempo4 = toc(tempo3);


clc
kTrain
kTest
res1 = [trainSens trainSpec; testSens testSpec]
res2 = [PQtrainSens PQtrainSpec; PQtestSens PQtestSpec]

disp(['Time (random) ', num2str(tempo2+tElapsed), ' s'])
disp(['Time (circulant) ', num2str(tempo4+tElapsed), ' s'])

clear tElapsed logpH logpP q qs s TOL tStart v iRow i emisH emisP