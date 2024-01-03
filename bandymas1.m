clear all

exfile = readtable('BaselTest1.csv');
data = table2array(exfile(:,1:2));

%data2 = data(1:7:end,:);

YearCol = nan(length(data),1);
MonthCol = nan(length(data),1);
DayCol = nan(length(data),1);
HourCol = nan(length(data),1);

data = [data YearCol MonthCol DayCol HourCol];

for i=1:length(data)
    date = string(data(i,1));
    year = extractBetween(date,1,4);
    month = extractBetween(date,5,6);
    day = extractBetween(date,7,8);
    hour = extractBetween(date,9,10);

    data(i,3) = year;
    data(i,4) = month;
    data(i,5) = day;
    data(i,6) = hour;
end

requiredData = data(data(:,4) == 3 & data(:,5) >=7 & data(:,5) <= 21 & ((data(:,6) >= 0 & data(:,6) <= 3) | data(:,6) == 6),2);

input1 = requiredData(1:5:end,:);
input2 = requiredData(2:5:end,:);
input3 = requiredData(3:5:end,:);
input4 = requiredData(4:5:end,:);
output = requiredData(5:5:end,:);

%input1 = input1(:,2);
%input2 = input2(:,2);
%input3 = input3(:,2);
%input4 = input4(:,2);
%output = output(:,2);

preparedData = [input1 input2 input3 input4 output];

train_ratio = 0.784; % 80% of the data for training, 20% for validation
train_size = round(train_ratio * length(preparedData));

testData = preparedData(train_size+1:end,:);

for i=1:length(testData)
    error = abs(testData(i,4) - testData(i,5));
    allErrors(i,1) = error;
end

min(allErrors)
max(allErrors)
mean(allErrors)
std(allErrors)
mse = mean((testData(:,4) - testData(:,5)).^2);
disp(['Mean Squared Error: ', num2str(mse)]);