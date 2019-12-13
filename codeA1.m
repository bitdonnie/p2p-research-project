% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'LoanData_.csv');
delimiterIn = ',';
headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
T = readtable(filename);

descriptiveVars = ["Age", "BidsPortfolioManager", "BidsApi", "BidsManual", "AppliedAmount", "Amount", "Interest", "LoanDuration", "MonthlyPayment", "IncomeFromPrincipalEmployer", "IncomeFromPension", "IncomeFromSocialWelfare", "IncomeFromLeavePay", "IncomeFromChildSupport", "IncomeOther", "IncomeTotal", "LiabilitiesTotal", "DebtToIncome", "FreeCash", "LossGivenDefault", "ExpectedReturn", "ProbabilityOfDefault", "PlannedPrincipalPostDefault", "PlannedPrincipalTillDate", "PrincipalOverdueBySchedule", "PlannedInterestPostDefault", "EAD1", "EAD2", "PrincipalRecovery", "InterestRecovery", "PrincipalPaymentsMade", "InterestAndPenaltyPaymentsMade", "PrincipalBalance", "InterestAndPenaltyBalance", "NoOfPreviousLoansBeforeLoan", "AmountOfPreviousLoansBeforeLoan", "PreviousRepaymentsBeforeLoan", "PreviousEarlyRepaymentsBefoleLoan", "InterestAndPenaltyDebtServicingCost", "PrincipalDebtServicingCost"];
descriptiveTable = table();

for i = 1:length(descriptiveVars)
	descriptiveTable.Variable(i) = descriptiveVars(i);

	% Clean the vector to remove NaN obs.
	cleanVector = T.(descriptiveVars(i))(~isnan(T.(descriptiveVars(i))));
	descriptiveTable.Obs(i) = length(cleanVector);
	descriptiveTable.Mean(i) = mean(cleanVector);
	descriptiveTable.Max(i) = max(cleanVector);
	descriptiveTable.Min(i) = min(cleanVector);
	descriptiveTable.Std(i) = std(cleanVector);
end

% write table to excel to format for Latex
writetable(descriptiveTable, strcat(rootFilePath, 'tables/DescriptivesTable.xlsx'));

useOfLoan = T.UseOfLoan;
categories = categorical(useOfLoan, [0, 1, 2, 3, 4, 5, 6, 7, 8], {'Loan consolidation', 'Real estate', 'Home improvement', 'Business', 'Education', 'Travel', 'Vehicle', 'Other', 'Health'});
histogram(categories);

education = T.Education;
categories = categorical(education, [1, 2, 3, 4, 5], {'Primary', 'Basic', 'Vocational', 'Secondary', 'Higher'});
histogram(categories);

maritalStatus = T.MaritalStatus;
categories = categorical(maritalStatus, [1, 2, 3, 4, 5], {'Married', 'Cohabitant', 'Single', 'Divorced', 'Widow'});
histogram(categories);

employmentStatus = T.EmploymentStatus;
categories = categorical(employmentStatus, [1, 2, 3, 4, 5, 6], {'Unemployed', 'Partially employed', 'Fully employed', 'Self-employed', 'Entrepreneur', 'Retiree'});
histogram(categories);

% convert NrOfDependants to a int
nrOfDependants = str2double(T.NrOfDependants);
histogram(nrOfDependants);

% Convert status to double to create histogram
loanStatus = zeros(length(T.Status), 1);
for i = 1:length(T.Status)
	if string(T.Status(i)) == 'Repaid'
		loanStatus(i) = 1;
	elseif string(T.Status(i)) == 'Late'
		loanStatus(i) = 2;
	else
		loanStatus(i) = 3;
	end
end
categories = categorical(loanStatus, [1, 2, 3], {'Repaid', 'Late', 'Default'});
histogram(categories);
