close all;
clear;

load('mnist-testing.mat');
load('mnist-training.mat');

%% Part i

% Finds the first nine examples of each digit in the training sample and
% stores them all in first9EachDigit by column
first9EachDigit = zeros(90,1);
for i = 0:9
    first9EachDigit(i*9+1:i*9+9) = find(trainLabels == i, 9);
end

% Plots each 9 examples into their own plot for each digit
for i = 0:9
    f = figure();
    hold on;
    for j = 1:3
        for k = 1:3
            index = first9EachDigit(i*9 + (j-1)*3 + k);
            subplot(3, 3, (j-1)*3 + k);
            imagesc(trainImages(:, :, index));
            colormap('gray');
            axis square;
            axis off;
        end
    end
    subplot(3, 3, 2);
    colorbar('Position', [0.93, 0.11, 0.02, .82]);
    title("9 Example Images of Digit " + num2str(i));
    hold off;
end

%% Part ii

% flattens the images into vectors from matrices
flatTrainImages = zeros(784,24000);
for i = 1:24000
    flatImage = zeros(28,1);
    for col = 1:28
        flatImage((col-1)*28+1:col*28) = trainImages(:, col, i);
    end
    flatTrainImages(:, i) = flatImage;
end

%% Part iii a

% Number of samples
n = 500;

% gets the number of samples from the flatTrainImages matrix and their
% corresponding labels
X = flatTrainImages(:,n);
Xlabel = trainLabels(n);

% calculates the psuedoInverse for and stores it in w
psuedoInverse = pinv(X);
w = psuedoInverse'*Xlabel;

% Make the guesses with the psuedoInverse
predictedY = round(flatTrainImages' * w);

% Plots the result using confusionchart()
figure();
confusionchart(trainLabels, predictedY);
title("Confusion Chart for " + num2str(n) + " Training Images");
xlabel("Predicted Digit");
ylabel("True Digit");

%% Part iii b 
% Self Made plot to show that confusionchart() does a very similar thing 
% with less code and is a viable way to plot the results of our guesses
confusionMatrix = zeros(10, 10);
outOfBounds = 0;
% make the confusion matrix and display it in the Command Window
for i = 1:length(trainLabels)
    trueClass = trainLabels(i) + 1;
    predictedClass = round(predictedY(i)) + 1;

    % If the guess out of bounds store it in a counter
    if trueClass > 10 || predictedClass > 10
    outOfBounds = outOfBounds + 1;
    else
    confusionMatrix(trueClass, predictedClass) = confusionMatrix(trueClass, predictedClass) + 1;
    end
end

disp('Confusion Matrix:');
disp(confusionMatrix);

% Single out the correct guesses to plot them more distinctly
correctConfusionMatrix = zeros(10,10);
for i = 1:10
    correctConfusionMatrix(i,i) = confusionMatrix(i,i);
    confusionMatrix(i,i) = 0;
end

figure();
hold on;

% This plotting of two imagesc seems to completely disregard the values in
% the confusionMatrix plot and set them all to the base value of the
% colormap, the correctConfusionMatrix is graphs with the correct coloring
imagesc(confusionMatrix);
imagesc(correctConfusionMatrix);

% color map from light orange (1,.65,.53) to a blue (1,.5,.8) in rgb
betterColorMap = [linspace(1,.1)', linspace(.65,.5)', linspace(.53,.8)'];

% use the custom colormap
colormap(betterColorMap);

% Visual changes
axis([.5 10.5 .5 10.5]);
title("Confusion Matrix for " + num2str(n) + " Training Images");
xlabel("Predicted Digit");
ylabel("True Digit");
xticklabels(0:9);
yticklabels(0:9);

% Writes the numbers into each section with a white font color
for i = 1:10
    for j = 1:10
        if j == i
        else
        text(j, i, num2str(confusionMatrix(i, j)), ...
            "HorizontalAlignment", "center", ...
            "Color", "White");
        yline(j-.5);
        end
    end
    xline(i-.5);
end

% Writes the numbers into the color diagonal with a black font color to
% distinguish it from the wrong guesses.
for i = 1:10
    text(i, i, num2str(correctConfusionMatrix(i, i)), ...
        "HorizontalAlignment", "center");
end

hold off;

% print the number of outOfBounds errors if they exist
if outOfBounds > 0
    fprintf("Guesses Out Of Bounds: %i", outOfBounds);
end
