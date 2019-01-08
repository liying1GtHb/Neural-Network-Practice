% adapted from the python program in Neural Network and Deep Learning 
% by Michael Nielson;

% Properties of a Neural Network class:
% Sizes: a row vector that gives the number of neurons in each layer;
% NumLayers: number of layers not counting the input layer;
% Weights and Biases: as name suggested;
% Cost: cost function. The following are the options.
% 'Quadratic', 'CrossEntropy','SoftMax';
% Regularization: the options are 'L1','L2', and 'DropOut'. If no
% regularization is preferred, use either 'L1' or 'L2' with lambda=0;
% lambda: regularization parameter for 'L1' or 'L2'. If 
% regularization='Dropout', set lambda=0;

% In the cost function, use weight decay (L2 regularization);

% To fit a function, such as sin(x) using neural network, run, for example,
% net = SimpleNeuralNetworkYL([1 100 100 1],'CrossEntropy','L1',1);
% net.SGDFit(trainingX,trainingY,epochs,minibat,eta,lambda);
% Then evaluate the NN at a set of input:
% y = net.forward(evalX);

% To classify, run, for example,
% net = SimpleNeuralNetworkYL([784 30 10],'Quadratic','L2',5);
% net.SGDClf(trainingX,trainingY,epochs,minibat,eta,testX,testY+1)
% Note that testY gives the digits, whereas testY+1 gives the indices of
% ones in the 10-dimensional vectors;
% 
classdef SimpleNeuralNetworkYL < handle
    properties
        Sizes
        NumLayers
        Weights
        Biases
        Cost
        Regularization
        Lambda
    end
    methods
        function obj = SimpleNeuralNetworkYL(sizes,cost,regularization,lambda)
            obj.Sizes = sizes;
            obj.NumLayers = length(sizes)-1;
            weights = cell(1,obj.NumLayers);
            for i = 1:obj.NumLayers
                weights{i} = randn(obj.Sizes(i+1),obj.Sizes(i));
            end
            obj.Weights = weights;
            biases = cell(1,obj.NumLayers);
            for i = 1:obj.NumLayers
                biases{i} = randn(obj.Sizes(i+1),1);
            end
            obj.Biases = biases;
            obj.Cost = cost;
            obj.Regularization = regularization;
            obj.Lambda =lambda;
        end 
        function a = feedForward(obj,x)
            a = x;
            for k = 1:obj.NumLayers
                z = obj.Weights{k}*a+obj.Biases{k};
                a = obj.sigmoid(z);
            end
            if strcmp(obj.Cost,'SoftMax') == 1
                a = exp(z)./sum(exp(z));
            end
        end

        function SGDFit(obj,trainingX,trainingY,epochs,minibat,eta)
            % Stochastic Gradient Descent method for fitting problems; for
            % example, fit y=sin(x) using a neural network;
            % trainingX: the input matrix of training data. Its number of 
            % columns is the amount of training data; its number of rows is
            % the number of components of each input.
            % trainingY: the output matrix of training data. The format is the
            % same as trainingX;
            % epochs: number of epochs to train;
            % minibat: number of data in a minibatch;
            % eta: learning rate;
            % lambda: regularization parameter;
        
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
                    obj.updateMinibat(dataX,dataY,numData,eta);
                end
                fprintf('Epoch %d complete. \n', i);
            end
        end
        function SGDClf(obj,trainingX,trainingY,epochs,minibat,eta,testX,testY)
            % Stochastic Gradient Descent method for classification problems; 
            % for example, the digit recognization problem using NN;
            % trainingX: the input matrix of training data. Its number of 
            % columns is the amount of training data; its number of rows is
            % the number of components of each input.
            % trainingY: the output matrix of training data. The format is the
            % same as trainingX;
            % epochs: number of epochs to train;
            % minibat: number of data in a minibatch;
            % eta: learning rate;
            % testX,testY: the same format as trainingX. These are test
            % data to check the correctness rate of the NN;
            % lambda: regularization parameter;
        
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
                    obj.updateMinibat(dataX,dataY,numData,eta);
                end
                % In this epoch, check how many are correct in test data;
                y = obj.feedForward(testX);
                numtestData = size(testX,2);
                [~,indy] = max(y,[],1);
                numCorr = sum(indy==testY);
                fprintf('Epoch %d %d correct out of %d.\n', i, numCorr, numtestData); 
            end
        end
        function updateMinibat(obj,dataX,dataY,numData,eta)
            % for each minibatch, update weights and biases using eta;
            % dataX,dataY: minibatch of input and output;
            % eta: learning rate;
            minibat = size(dataX,2);
            [sumdeltab,sumdeltaw]=obj.backProp(dataX,dataY);
            % backward pass
            if strcmp(obj.Regularization,'L1') == 1
                wRegu = cellfun(@minus,obj.Weights,cellfun(@(x)x*eta*obj.Lambda/numData,cellfun(@sign,obj.Weights,'un',0),'un',0),'un',0);
                obj.Weights = cellfun(@minus,wRegu,cellfun(@(x)x*eta/minibat,sumdeltaw,'un',0),'un',0);
            else
                obj.Weights = cellfun(@minus,cellfun(@(x)x*(1-eta*obj.Lambda/numData),obj.Weights,'un',0),cellfun(@(x)x*eta/minibat,sumdeltaw,'un',0),'un',0);
            end
            obj.Biases = cellfun(@minus,obj.Biases,cellfun(@(x)x*eta/minibat,sumdeltab,'un',0),'un',0);      
        end
        function [db,dw] = backProp(obj,dataX,dataY)
            % Back propagation procedure to update weights and biases;
            
            % feedforward;
            
            % z: z=w*a+b, a=activation(z) in each layer;
            % zs, as: store z and a values for all layers;
            % Note that the numbers of cells of zs and as are both 1 more than 
            % the numbers of cells of weights and biases;
            if strcmp(obj.Regularization,'DropOut') == 1
                tempWeights = obj.Weights;
                for k = 1:obj.NumLayers-1
                    m = obj.Sizes(k+1);
                    tempperm = randperm(m,floor(m/2));
                    tempWeights{k}(tempperm,:) = 0;
                    tempWeights{k+1}(:,tempperm) = 0;
                end
            end
            z = dataX;
            a = z;  
            zs = cell(1,obj.NumLayers+1);
            as = cell(1,obj.NumLayers+1);
            zs{1} = z;
            as{1} = a;
            for k = 1:obj.NumLayers
                if strcmp(obj.Regularization,'DropOut') == 1
                    z = tempWeights{k}*a+obj.Biases{k};
                else
                    z = obj.Weights{k}*a+obj.Biases{k};
                end
                zs{k+1} = z;
                a = obj.sigmoid(z);
                as{k+1} = a;
            end
            y = dataY;
            switch obj.Cost
                case 'Quadratic'
                    delta = (as{obj.NumLayers+1}-y).*obj.sigmoidprime(z);
                case 'CrossEntropy'
                    delta = (as{obj.NumLayers+1}-y);
                case 'SoftMax'
                    as{obj.NumLayers+1} = exp(z)./sum(exp(z));
                    delta = (as{obj.NumLayers+1}-y);
            end
            deltab = cell(1,obj.NumLayers);
            deltaw = cell(1,obj.NumLayers);
            deltab{obj.NumLayers} = delta;
            deltaw{obj.NumLayers} = delta*as{obj.NumLayers}';

            for k = obj.NumLayers-1:-1:1
                z = zs{k+1};
                if strcmp(obj.Regularization,'DropOut') == 1
                    delta = (tempWeights{k+1}'*delta).*obj.sigmoidprime(z);
                else    
                    delta = (obj.Weights{k+1}'*delta).*obj.sigmoidprime(z);
                end
                deltab{k} = delta;
                deltaw{k} = delta*(as{k}');
            end

            db = cellfun(@(x)sum(x,2),deltab,'un',0);
            dw = deltaw;            
        end
    end
    methods(Static)
        function z = sigmoid(x)
            z = 1./(1+exp(-x));
        end
        function zp = sigmoidprime(x)
           % cannot use exp(-z)./((1+exp(-z)).^2); causes infinity
           % divided by infinity when z is a big negative number;
           zp = NeuralNetwork.sigmoid(x).*(1-NeuralNetwork.sigmoid(x));
        end
    end
end
