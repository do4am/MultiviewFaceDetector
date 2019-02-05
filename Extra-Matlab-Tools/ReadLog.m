format long g
A = csvread('trainlog_3_test.txt',2,0);
y = A(:,4);
x = A(:,1);
B = csvread('trainlog_3_train.txt',2,0);
x2 = B(1:11:end,1);
y2 = B(1:11:end,4);

range = 0:0.01:55000;
y_p = polyval(polyfit(x,y,15),range);
y2_p = polyval(polyfit(x2,y2,15),range);

y_chip = spline(x,y,range);
y2_chip = min(spline(x2,y2,range),1);
%y2_chip = max(y2_chip - (max(y2_chip)-1),0);
%figure()
ax1 = subplot(2,1,1);
hold on
plot(range, y_chip, 'b', 'LineWidth',1)
plot(range, y2_chip, 'r', 'LineWidth',1)

xlabel('Iteration')
ylabel('Accuracy')
legend('Test-Accuracy', 'Train-Accuracy', 'Location','SouthEast')

ax2 = subplot(2,1,2);
hold on
plot(range, y_p, 'b', 'LineWidth',1)
plot(range, y2_p, 'r', 'LineWidth',1)

xlabel('Iteration')
ylabel('Accuracy')
legend('Test-Accuracy-Average', 'Train-Accuracy-Average', 'Location','SouthEast')

title('Accuracy of Train and Test over 55000 Iterations (approximately 10 epochs)')
