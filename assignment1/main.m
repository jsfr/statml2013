graphics_toolkit('gnuplot');

% I.2.1
function y = gauss(x,m,s)
    y = 1 / sqrt(2 * pi * s^2) * exp(- 1/(2*s^2) * (x - m)^2);
end

function h = proper_plot(X, Y)
    h = figure(1);
    plot(X, Y, 'r-', 'LineWidth', 4);
    grid on;
    axis ([X(1), X(end), 0, max(Y)]);
    W = 4; H = 3;
    set(h,'PaperUnits','inches');
    set(h,'PaperOrientation','portrait');
    set(h,'PaperSize',[H,W]);
    set(h,'PaperPosition',[0,0,W,H]);
    FS = findall(h,'-property','FontWeight');
    set(FS,'FontWeight','bold');
    % pause
end

X = [-5:0.01:3];
h = proper_plot(X, arrayfun(@(x) gauss(x,-1,1), X));
print(h,'-dpng','-color','I21_1.png');

X = [-8:0.01:8];
h = proper_plot(X, arrayfun(@(x) gauss(x,0,2), X));
print(h,'-dpng','-color','I21_2.png');

X = [-10:0.01:14];
h = proper_plot(X, arrayfun(@(x) gauss(x,2,3), X));
print(h,'-dpng','-color','I21_3.png');

% I.2.2
n = 100;
u = [1 2]';
CovMat = [0.3 0.2; 0.2 0.2];
R = randn(n,2);

function Y = sample_gauss(R, u, CovMat)
    L = chol(CovMat, 'lower');
    Y = zeros(size(R, 1),2);
    for k=1:size(R, 1)
        Y(k,:) = u + L * R(k,:)';
    end
end

Y = sample_gauss(R, u, CovMat);
h = figure(1);
plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','r');
grid on;
print(h,'-dpng','-color','I22_1.png');

% I.2.3
hold on;
su = mean(Y);
plot(u(1), u(2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','b');
plot(su(1), su(2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','g');
grid on;
print(h,'-dpng','-color','I23_1.png');
hold off;

distanceMean = abs(u - su')

% I.2.4
SCovMat = zeros(2, 2);
for k=1:n
    t = Y(k,:)' - su';
    SCovMat = SCovMat + t * t';
end
SCovMat = SCovMat / n
% TODO: These seem arbitrary and wrong, recheck the formulas later!

[EigenMat, eigenValues] = eig(SCovMat)
EigenMat(:,1) = u + sqrt(eigenValues(1,1)) * EigenMat(:,1)
EigenMat(:,2) = u + sqrt(eigenValues(2,2)) * EigenMat(:,2)

plot(EigenMat(1,:), EigenMat(2,:), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','r');
grid on;

function result = rotate_matrix(mat, degree)
    rad = degree * pi/180;
    rmat = [cos(rad) -sin(rad); sin(rad) cos(rad)];
    result = inv(rmat) * mat * rmat;
end

CovMat30 = rotate_matrix(SCovMat, 30);
Y = sample_gauss(R, u, CovMat30);
h = figure(1);
plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','r');
hold on;

CovMat60 = rotate_matrix(SCovMat, 60);
Y = sample_gauss(R, u, CovMat60);
plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','g');

CovMat90 = rotate_matrix(SCovMat, 90);
Y = sample_gauss(R, u, CovMat90);
plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','b');
grid on;
print(h,'-dpng','-color','I24_1.png');
hold off;

% TODO: This part seems to give a wrong rotation, check later
% Y = sample_gauss(R, u, SCovMat);
% t = polyfit(Y(:, 1), Y(:, 2), 1)

% plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','r');
% hold on;
% plot([min(Y(:,1)):max(Y(:,1))], arrayfun(@(x) t(1)*x+t(2), [min(Y(:,1)):max(Y(:,1))]), 'r-', 'LineWidth', 4);

% dataAngle = acos(dot([1 t(1)]', [1 0]')/(norm([1 t(1)]')*norm([1 0]'))) * 180/pi
% Y = sample_gauss(R, u, rotate_matrix(SCovMat, dataAngle));
% t = polyfit(Y(:, 1), Y(:, 2), 1)
% dataAngle2 = acos(dot([1 t(1)]', [1 0]')/(norm([1 t(1)]')*norm([1 0]'))) * 180/pi
% plot(Y(:,1), Y(:,2), 'ok', 'MarkerSize', 10, 'MarkerFaceColor','b');
% hold on;
% plot([min(Y(:,1)):max(Y(:,1))], arrayfun(@(x) t(1)*x+t(2), [min(Y(:,1)):max(Y(:,1))]), 'b-', 'LineWidth', 4);
% W = 4; H = 3;
% set(h,'PaperUnits','inches');
% set(h,'PaperOrientation','portrait');
% set(h,'PaperSize',[H,W]);
% set(h,'PaperPosition',[0,0,W,H]);
% FS = findall(h,'-property','FontWeight');
% set(FS,'FontWeight','bold');
% axis ([-2, 4, -2, 4]);
% print(h,'-dpng','-color','I24_2.png');

pause