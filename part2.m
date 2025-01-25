close all;
clear;

load('mnist-testing.mat');
load('mnist-training.mat');

%% Flattening
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

%% Part 2 i

% Initialize vectors to store errors for each digit's guesses
errors1 = ones(10,1);
errors2 = ones(10,1);

% loop through digits
for k = 0:9

% Isolate the values that correspond to the current digit
trainKLabels = trainLabels(:,1) == k;
trainKImages = flatTrainImages(:,trainKLabels);
trainKLabels = ones(length(trainKImages(1,:)),1);
modifiedTestLabels = -1 * ones(8000,1);
modifiedTestLabels(testLabels == k) = 1;

% Create a bounds vector that is true where an image of the current digit
% has a value > 0 for a pixel
bounds = false(784,1);
for i = 1:size(trainKImages, 2)
    for j = 1:784
        if trainKImages(j,i) > 0
            bounds(j) = true;
        end
    end
end

% Create a simple binary classifier based on the bounds
w = zeros(784,1);
w(bounds) = 1;
w(~bounds) = -1;

%% Part 2 ii

% Uses the simple binary classifier to predict if the digit is in the
% current class or not
predictions = zeros(length(trainKImages(1,:)),1);
for i = 1:length(trainKImages(1,:))
    f = w' * trainKImages(:,i);
    if f >= 0
        predictions(i) = 1;
    else
        predictions(i) = -1;
    end
end

% Calculates the error for the current class
figure();
cm = confusionchart(double(trainKLabels), predictions);
title(['Confusion Matrix for Digit ', num2str(k)]);

correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
errorRate = (totalPredictions - correctPredictions) / totalPredictions;
disp(['Error Rate for class ', num2str(k), ': ', num2str(errorRate)]);
errors1(k+1) = errorRate;

% The classifier on its own class should have no error, only plots erroneous
% results
if errorRate == 0
    close;
end

% Use the simple binary classifier to predict if the flatTestImages are in
% the current class or not
predictions = zeros(length(trainKImages(1,:)),1);
for i = 1:length(flatTestImages(1,:))
    f = w' * flatTestImages(:,i);
    if f >= 0
        predictions(i) = 1;
    else
        predictions(i) = -1;
    end
end

% Plots the simple binary classifier versus flatTestImages confusion matrix
figure();
cm = confusionchart(modifiedTestLabels, predictions);
title(['Confusion Matrix for Digit ', num2str(k)]);

% Calculates and stores errors
correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
errorRate = (totalPredictions - correctPredictions) / totalPredictions;
disp(['Error Rate for testImages for ', num2str(k), ': ', num2str(errorRate)]);
errors2(k+1) = errorRate;
end

% Calculate average error rates
avgErrors = mean(errors1);

% Display the average error rates
disp(['Average Error Rate for Part 2 ii for class "k": ', num2str(avgErrors)]);

% Calculate average error rates
avgErrors = mean(errors2);

% Display the average error rates
disp(['Average Error Rate for Part 2 ii for testImages: ', num2str(avgErrors)]);

%% Part 2 iii a

for k = 0:9
trainKLabels = trainLabels(:,1) == k;
trainKImages = flatTrainImages(:,trainKLabels);
trainKLabels = ones(length(trainKImages(1,:)),1);
modifiedTestLabels = -1 * ones(8000,1);
modifiedTestLabels(testLabels == k) = 1;

% Instead of a bounds, determine the frequency of the pixel's appearence
% accross images
w = zeros(784,1);
for i = 1:size(trainKImages, 2)
    for j = 1:784
        if trainKImages(j,i) > 0
            w(j) = w(j) + 1;
        end
    end
end

% Sets more a more refined binary classifier that only positive when the
% pixel is prevalent in many images for the current digit
threshold = length(trainKImages(:,1)) / 5;
w(0<w & w<threshold) = -1;
w(w>=threshold) = 1;

% Makes predictions using new inary classifier for the current class on the
% current class
predictions = zeros(length(trainKImages(1,:)),1);
for i = 1:length(trainKImages(1,:))
    f = w' * trainKImages(:,i);
    if f >= 0
        predictions(i) = 1;
    else
        predictions(i) = -1;
    end
end

% Get confusion matrix
figure();
cm = confusionchart(double(trainKLabels), predictions);
title(['Confusion Matrix for Digit ', num2str(k)]);

% Calculates error and stores in vecotr
correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
errorRate = (totalPredictions - correctPredictions) / totalPredictions;
disp(['Error Rate for class ', num2str(k), ': ', num2str(errorRate)]);
errors1(k+1) = errorRate;

% The classifier on its own class should have no error, only plots erroneous
% results
if errorRate == 0
    close;
end

% Makes predictions using the new binary classifier on flatTestImages
predictions = zeros(length(trainKImages(1,:)),1);
for i = 1:length(flatTestImages(1,:))
    f = w' * flatTestImages(:,i);
    if f >= 0
        predictions(i) = 1;
    else
        predictions(i) = -1;
    end
end

% Plots the confusion matrix for the testImages and predictions using the
% new binary classifier
figure();
cm = confusionchart(modifiedTestLabels, predictions);
title(['Confusion Matrix for Digit ', num2str(k)]);

correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
errorRate = (totalPredictions - correctPredictions) / totalPredictions;
disp(['Error Rate for testImages for ', num2str(k), ': ', num2str(errorRate)]);
errors2(k+1) = errorRate;
end

% Calculate average error rates
avgErrors = mean(errors1);

% Display the average error rates
disp(['Average Error Rate for Part 2 iii a for class "k": ', num2str(avgErrors)]);

% Calculate average error rates
avgErrors = mean(errors2);

% Display the average error rates
disp(['Average Error Rate for Part 2 iii a for testImages: ', num2str(avgErrors)]);


%% Part 2 iii b

errors = zeros(10,1);

% loop through all the digits
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
weights = trainLabelsBinary * pseudoinverse;

% This matrix contains the predicted digits for each image.
% It performs a linear transformation of the input images using the learned
% weights. The result is a raw score for the current digit. The sign 
% function is used here to convert the raw scores into binary outputs 
predictions = sign(weights * flatTestImages);

% Create confusion matrix for evaluation
figure();
cm = confusionchart(testLabelsBinary, predictions);
title(['Confusion Matrix for Digit ', num2str(i)]);
xlabel("Predicted Digit");
ylabel("True Digit");

% Calculate the error
correctPredictions = sum(diag(cm.NormalizedValues));
totalPredictions = sum(cm.NormalizedValues, 'all');
errorRate = (totalPredictions - correctPredictions) / totalPredictions;
disp(['Error Rate for ', num2str(i), ': ', num2str(errorRate)]);
errors(i+1) = errorRate;
end

% Calculate average error rates
avgErrors = mean(errors);

% Display the average error rates
disp(['Average Error Rate for Part 2 iii b testImages: ', num2str(avgErrors)]);
