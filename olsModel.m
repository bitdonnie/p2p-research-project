% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'latestData.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);
y = T.m_yield;

% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'default', 'm_yield'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(length(y), 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = y(idxTrain);
XTest = X(idxTest,:);
yTest = y(idxTest);

% run Lasso Binominal Logistic Regression
[B, FitInfo] = lasso(XTrain, yTrain, 'CV', 10, 'PredictorNames', T.Properties.VariableNames);

% use minimum standard error
idxLamdaMinDeviance = FitInfo.IndexMinMSE;
coef = B(:, idxLamdaMinDeviance);
coef0 = FitInfo.Intercept(idxLamdaMinDeviance);

% use with 1 standard error minimization
% idxLambda1SE = FitInfo.IndexMin1SE;
% coef = B(:, idxLambda1SE);
% coef0 = FitInfo.Intercept(idxLambda1SE);

sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLamdaMinDeviance)~=0)

% lassoPlot(B, FitInfo, 'PlotType', 'CV');
% lengend('show')

yhat = XTest * coef + coef0;

hold on
scatter(yTest,yhat)
plot(yTest,yTest)
xlabel('Yearly Yield')
ylabel('Predicted Yearly Yield')
hold off
export_fig(strcat(rootFilePath, 'Figures/', 'LASSOPredictions_PCA.png'))

% regular linear regression model with only one predictor
mdl = fitlm(XTest(:, 2), yTest);
y_predictions = predict(mdl, XTest(:, 2));

hold on
scatter(yTest, y_predictions)
plot(yTest,yTest)
xlabel('Actual Monthly Yield')
ylabel('Predicted Monthly Yield')
hold off

% write LASSO models to xlxs
predictorVars = table(transpose(FitInfo.PredictorNames(B(:,idxLamdaMinDeviance)~=0)), nonzeros(B(:,idxLamdaMinDeviance)));
writetable(predictorVars, strcat(rootFilePath, 'Tables/LASSOGLRTable.xlsx'));
