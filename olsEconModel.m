import export_fig.*

% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'latestData.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
y = T.m_yield;

% remove irrelevant variables from table
relVariables = ["age", "interest", "loanduration", "existingliabilities", "m_newcreditcustomer", "verificationtype3", "verificationtype4", "gender2", "gender3", "country2", "country3", "country4", "useofloan2", "useofloan3", "useofloan4", "useofloan5", "useofloan6", "useofloan7", "useofloan8", "useofloan9", "education2", "education3", "education4", "education5", "employmentstatus1", "homeownershiptype1", "log_incometotal", "log_appliedamount", "log_liabilitiestotal"];
tableProps = T.Properties.VariableNames;
for i = 1:length(tableProps)
    if ~any(contains(relVariables, string(tableProps(i))))
        T = removevars(T, tableProps(i));
    end
end

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

mdl = fitlm(XTrain, yTrain);

yhat = predict(mdl, XTest);

hold on
scatter(yTest,yhat)
plot(yTest,yTest)
xlabel('Yearly Yield')
ylabel('Predicted Yearly Yield')
hold off
export_fig(strcat(rootFilePath, 'Figures/', 'OLSPredictions_Econ.png'))

% write logit model to xlxs 
mdl = fitlm(X, y, 'linear', ...
	 'VarNames', cat(1, T.Properties.VariableNames{:}, {'default'}));
writetable(mdl.Coefficients, strcat(rootFilePath, 'Tables/EcoOLSTable.xlsx'));
