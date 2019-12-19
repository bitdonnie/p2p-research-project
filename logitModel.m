% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'latestData.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
obsDefault = T.default;
% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'default', 'm_yield'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(obsDefault, 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = obsDefault(idxTrain);
XTest = X(idxTest,:);
yTest = obsDefault(idxTest);

% run Lasso Binominal Logistic Regression
[B, FitInfo] = lassoglm(XTrain, yTrain, 'binomial', 'CV', 10, 'PredictorNames', T.Properties.VariableNames);

idxLamdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLamdaMinDeviance);
coef = [B0; B(:, idxLamdaMinDeviance)];

% idxLamda1SE = FitInfo.Index1SE;
% B0 = FitInfo.Intercept(idxLamda1SE);
% coef = [B0; B(:, idxLamda1SE)];

% lassoPlot(A, FitInfo, 'plottype', 'CV');
yhat = glmval(coef, XTest, 'logit');
yhatBinom = (yhat>=0.5);
yTest = (yTest==1);

c = confusionchart(yTest,yhatBinom);
export_fig(strcat(rootFilePath, 'Figures/', 'ConfusionMatrix_LASSO.png'))

accuracy = (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 2)) / (c.NormalizedValues(1, 1) + c.NormalizedValues(1, 2) + c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
sensitivity = c.NormalizedValues(2, 2) / (c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
specifity = c.NormalizedValues(1, 1) / (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 1))

% Compute Area under the curve
[X, Y, T, AUC] = perfcurve(yTest, yhat, 1);
hold on
plot(X,Y)
plot([0, 1], [0, 1])
xlabel('False positive rate') 
ylabel('True positive rate')
title('LASSO for Classification by Binomial Logistic Regression')
hold off
export_fig(strcat(rootFilePath, 'Figures/', 'AUC_LASSO.png'))

% write LASSO models to xlxs
predictorVars = table(transpose(FitInfo.PredictorNames(B(:,idxLamdaMinDeviance)~=0)), nonzeros(B(:,idxLamdaMinDeviance)));
writetable(predictorVars, strcat(rootFilePath, 'Tables/LASSOLogitTable.xlsx'));

