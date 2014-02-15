h = figure(1);

% graphics_toolkit('gnuplot');

%%
% I.2.1
%
X = [-10:0.01:14];
Y1 = arrayfun(@(x) gauss(x,-1,1), X);
Y2 = arrayfun(@(x) gauss(x,0,2), X);
Y3 = arrayfun(@(x) gauss(x,2,3), X);

plot(X, Y1, 'r-', 'LineWidth', 1.5);
hold on;
plot(X, Y2, 'g-', 'LineWidth', 1.5);
plot(X, Y3, 'b-', 'LineWidth', 1.5);
hold off;
axis ([X(1), X(end), 0, max([Y1 Y2 Y3])]);
legend('[-1, 1]', '[0, 2]', '[2, 3]');
grid on;
print(h,'-dpng','I21.png');

%%
% I.2.2
%
n = 100;
mu = [1 2]';
Sigma = [0.3 0.2; 0.2 0.2];
R = randn(n,2);
Y1 = resampleGauss(R, mu, Sigma);

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','r');
grid on;
print(h,'-dpng','I22.png');

%%
% I.2.3
%
muML = mean(Y1)'
muDist = abs(mu - muML)

hold on; % Plot with the result from I22
plot(mu(1), mu(2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','b');
plot(muML(1), muML(2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','g');
hold off;
grid on;
print(h,'-dpng','I23.png');

%%
% I.2.4
%
SigmaML = zeros(2, 2);
for k=1:n
    t = Y1(k,:)' - muML;
    SigmaML = SigmaML + t * t';
end
SigmaML = SigmaML / n

[EigenVectors, eigenValues] = eig(SigmaML)

e1 = mu + sqrt(eigenValues(1,1)) * EigenVectors(:,1)
e2 = mu + sqrt(eigenValues(2,2)) * EigenVectors(:,2)

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','r');
hold on;
plot([muML(1) e1(1)], [muML(2) e1(2)], 'b-', 'LineWidth', 1.5);
plot([muML(1) e2(1)], [muML(2) e2(2)], 'b-', 'LineWidth', 1.5);
grid on;
print(h,'-dpng','I24_1.png');
hold off;

Sigma30 = rotateMatrix(SigmaML, 30);
Sigma60 = rotateMatrix(SigmaML, 60);
Sigma90 = rotateMatrix(SigmaML, 90);

Y2 = resampleGauss(R, mu, Sigma30);
Y3 = resampleGauss(R, mu, Sigma60);
Y4 = resampleGauss(R, mu, Sigma90);

plot(Y2(:,1), Y2(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','r');
hold on;
plot(Y3(:,1), Y3(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','g');
plot(Y4(:,1), Y4(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','b');
legend('30 deg', '60 deg', '90deg');
grid on;
print(h,'-dpng','I24_2.png');
hold off;


t = polyfit(Y1(:, 1), Y1(:, 2), 1)
dataAngle = acos(dot([1 t(1)]', [1 0]')/(norm([1 t(1)]')*norm([1 0]'))) * 180/pi
flippedEigen = fliplr(flipud(eigenValues));
Y5 = resampleGauss(R, mu, flippedEigen);
Y6 = resampleGauss(R, mu, rotateMatrix(SigmaML, dataAngle));
t2 = polyfit(Y5(:, 1), Y5(:, 2), 1)
t3 = polyfit(Y6(:, 1), Y6(:, 2), 1)

plot(Y1(:,1), Y1(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','r');
hold on;
plot(Y5(:,1), Y5(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','b');
plot(Y6(:,1), Y6(:,2), 'ok', 'MarkerSize', 3, 'MarkerFaceColor','g');
plot([-1:3], arrayfun(@(x) t(1)*x+t(2), [-1:3]), 'r-', 'LineWidth', 1.5);
plot([-1:3], arrayfun(@(x) t2(1)*x+t2(2), [-1:3]), 'b-', 'LineWidth', 1.5);
plot([-1:3], arrayfun(@(x) t3(1)*x+t3(2), [-1:3]), 'g-', 'LineWidth', 1.5);
grid on;
print(h,'-dpng','I24_3.png');

pause

%%
% I.4.1
%

%%
% I.4.2
%

%%
% I.4.3
%