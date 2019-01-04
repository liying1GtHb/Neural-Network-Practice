function [weights,biases,y] = NeuralNetworkMatYL(sizes,trainingX, trainingY, epochs, minibat, eta, choice, testX, testY)
% Implements neural network with training data;
% Change each minibatch to a matrix multipliction;

% Input parameters:
% sizes: a row vecotr, each component of which gives the number of nodes in
% the corresponding layer.
% trainingX: trainingX is the input matrix of training data. Its number of 
% columns is the amount of training data; its number of rows is the number 
% of components of each input. 
% trainingY, testX, testY: the same format of trainingX.
% epochs: number of epochs. If there is test data, after each epoch, use
% test data to produce an intermediate result. If there is no test data,
% each epoch is a complete sweep of the training data. The training data is
% shuffled for the next epoch.
% minibat: the amount of data in a minibatch, must divide the number of
% columns of trainingX and trainingY. A minibatch is a set of data to
% perform a single step of gradient descent. 
% eta: learning rate;
% choice: 1=fitting and predicting; 2=classification; we assume that for 1,
% testX will be given to predict corresponding output y; for 2, testX and
% testY will be given to check the correctness in each epoch and the output
% y=0;
% testX, testY: each a matrix, with each column an input/output data, and 
% number of columns the amount of input/output data; if both are present, 
% they should have the same numbers of columns;

% Output parameters:
% weights, biases: parameters fitting the neural network;
% y: =0 if choice=2; output with input testX if choice=1; it is a matrix 
% with each column the output of corresponding column of testX;


% initializing biases and weights. numLayer is the number of layers except
% the input layer. Note that the input layer does not need weight or biase.

numLayer = length(sizes)-1;
weights = cell(1,numLayer);
biases = cell(1,numLayer);
for i = 1:numLayer
    weights{i} = randn(sizes(i+1),sizes(i));
    biases{i} = randn(sizes(i+1),1);
end

% the amount of data;
numData = size(trainingX,2);
for i = 1:epochs
    % shuffle traning data for each epoch;
    tempperm = randperm(numData);
    trainingX = trainingX(:,tempperm);
    trainingY = trainingY(:,tempperm);
    
    % update weights and biases for each minibatch;
    for j = 1:(numData/minibat)
        dataX = trainingX(:,((j-1)*minibat+1):j*minibat);
        dataY = trainingY(:,((j-1)*minibat+1):j*minibat);
        
        % for each minibatch, update weights and biases using eta;
        sumdeltab = cell(1,numLayer);
        sumdeltaw = cell(1,numLayer);
        for m = 1:numLayer
            sumdeltab{m} = zeros(sizes(m+1),1);
            sumdeltaw{m} = zeros(sizes(m+1),sizes(m));
        end
        for n = 1:minibat
                  
            % feedforward;
            % z: z=w*a+b, a=activation(z) in each layer;
            % zs, as: store z and a values for all layers;
            % Note that the numbers of cells of zs and as are both 1 more than 
            % the numbers of cells of weights and biases;
            z = dataX(:,n);
            a = z;  
            zs = cell(1,numLayer+1);
            as = cell(1,numLayer+1);
            zs{1} = z;
            as{1} = a;
            for k = 1:numLayer
                z = weights{k}*a+biases{k};
                zs{k+1} = z;
                a = 1./(1+exp(-z));
                as{k+1} = a;
            end

            % backward pass
            y = dataY(:,n);
            s = 1./(1+exp(-z));
            sp = s.*(1-s);
            delta = (as{numLayer+1}-y).*sp;
            deltab = cell(1,numLayer);
            deltaw = cell(1,numLayer);
            deltab{numLayer} = delta;
            deltaw{numLayer} = delta*as{numLayer}';
            
            for k = numLayer-1:-1:1
                z = zs{k+1};
                % cannot use exp(-z)./((1+exp(-z)).^2); causes infinity
                % divided by infinity when z is a big negative number;
                s = 1./(1+exp(-z));
                sp = s.*(1-s);
                delta = (weights{k+1}'*delta).*sp;
                deltab{k} = delta;
                deltaw{k} = delta*(as{k}');
            end
            
            sumdeltab = cellfun(@plus,sumdeltab,deltab,'un',0);
            sumdeltaw = cellfun(@plus,sumdeltaw,deltaw,'un',0);
        end
        weights = cellfun(@minus,weights,cellfun(@(x)x*eta/minibat,sumdeltaw,'un',0),'un',0);
        biases = cellfun(@minus,biases,cellfun(@(x)x*eta/minibat,sumdeltab,'un',0),'un',0);
    end
    % if test data is provided, in this epoch, check how many are correct in test data;
    if choice == 2
        numData = size(testX,2);
        a = testX;
        for k = 1:numLayer
            z = weights{k}*a+biases{k};
            a = 1./(1+exp(-z));
        end
        y = a;
        [~,indy] = max(y,[],1);
        numCorr = sum(indy==testY);
        fprintf('Epoch %d %d correct out of %d.\n', i, numCorr, numData); 
    else
        fprintf('Epoch %d complete. \n', i);
    end
end
if choice == 1
    a = testX;
    for k = 1:numLayer
        z = weights{k}*a+biases{k};
        a = 1./(1+exp(-z));
    end
    y = a;
else
    y = 0;
end
end

