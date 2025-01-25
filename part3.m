close all;
clear;

load('mnist-testing.mat');
load('mnist-training.mat');

%% Part III

% Flatten images
flatTrainImages = zeros(784,24000);
flatTestImages = zeros(784,8000);
for i = 1:24000
    flatImage1 = zeros(28,1);
    flatImage2 = zeros(28,1);
    for col = 1:28
        flatImage1((col-1)*28+1:col*28) = trainImages(:, col, i);
        if i <= 8000
            flatImage2((col-1)*28+1:col*28) = testImages(:, col, i);
        end
    end
    flatTrainImages(:, i) = flatImage1;
    if i <= 8000
        flatTestImages(:, i) = flatImage2;
    end
end

% Slightly modified code from Part 2 iii, created a larger y matrix where
% each column corresponds to a digit and no longer calculates the error
% rate or plots the confusion matrix.
errors = ones(10,1);
weights = zeros(10,784);

% Loop through each digit
for i = 0:9

% Create digit labels: +1 for the target digit, -1 for other digits
testLabelsBinary = (double(testLabels == i) * 2 - 1)';
trainLabelsBinary = (double(trainLabels == i) * 2 - 1)';

% The pseudoinverse is used here to solve the least-squares problem for the 
% linear system weights * flatTestImages = testLabelsBinary.
% By computing the pseudoinverse, we find the solution that minimizes the 
% squared error between the predicted output and the actual labels, making 
% it efficient and direct.
pseudoinverse = pinv(flatTrainImages);

% This calculates the optimal weights by projecting the label matrix onto 
% the input via the pseudoinverse. Each row corresponds to a digit, and 
% each column corresponds to a pixel. This forms a linear relationship
% between input features and the target outputs.
weights(i+1,:) = trainLabelsBinary * pseudoinverse;
end

% digitClassifier function that takes in an image z and weights to
% determine what digit z the image z is most likely to be. This function
% flattens the image and multiplies the image by each of the weights for
% all the digits and the maximum value returned is the most likely digit
% and is the guess that is generated.
function k = digitClassifier(z,weights)
    flatZ = zeros(28,1);
    for col = 1:28
        flatZ((col-1)*28+1:col*28) = z(:, col);
    end
    guesses = ones(10,1) * -1;
    for i = 1:10
        guesses(i) = weights(i,:) * flatZ;
    end
    [~,k] = max(guesses);
    k = k-1;
end

% Variable to store the guessed digit
guessLabels = zeros(8000,1);

% Run the digitClassifier on all testImages stores guess in guessLabels
for i = 1:8000
    digit = testImages(:,:,i);
    guessLabels(i) = digitClassifier(digit,weights);
end

% Plot a confusion matrix
figure();
cm = confusionchart(testLabels,guessLabels);
title(['Confusion Matrix for ', num2str(i),' testImages']);
xlabel("Predicted Digit");
ylabel("True Digit");

% Calculate the error ((fp and fn) / total predictions)
correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
disp(correctPredictions);
disp(totalPredictions);
errorRate = (totalPredictions - correctPredictions) / totalPredictions;

% Display the average error rates
disp(['Error Rate for Part 3: ', num2str(errorRate)]);

