format long g
A = csvread('EvaluationTrainingLog_2702.train.txt',2,0);
y = A(:,5);
x = A(:,1);

B = csvread('EvaluationTrainingLog_train.txt',2,0);
y2 = B(:,5);
x2 = B(:,1);

range = 0:0.1:max(x);
y_chip = pchip(x,log(y),range);
y2_chip = pchip(x2,log(y2),range);

y_p = polyval(polyfit(x,y,1),range);
y2_p = polyval(polyfit(x2,y2,1),range);
%y2_chip = max(y2_chip - (max(y2_chip)-1),0);
%figure()
hold on
plot(range, y_p, 'b', 'LineWidth',1)
plot(range, y2_p, 'r', 'LineWidth',1)

xlabel('Iteration')
ylabel('LossRate')
legend('ImageNet-mean', 'Calculated-mean', 'Location','NorthEast')

title('LossRate')

