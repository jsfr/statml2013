graphics_toolkit('gnuplot');

function y = gauss(x,m,v)
    y = 1 / sqrt(2 * pi * v) * exp(- 1/(2*v) * (x - m)^2);
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
end

X = [-5:0.01:3];
h = proper_plot(X, arrayfun(@(x) gauss(x,-1,1), X), 'r-', 'LineWidth', 4);
print(h,'-dpng','-color','gauss1.png');

X = [-5:0.01:5];
h = proper_plot(X, arrayfun(@(x) gauss(x,0,2), X), 'r-', 'LineWidth', 4);
print(h,'-dpng','-color','gauss2.png');

X = [-6:0.01:10];
h = proper_plot(X, arrayfun(@(x) gauss(x,2,3), X), 'r-', 'LineWidth', 4);
print(h,'-dpng','-color','gauss3.png');