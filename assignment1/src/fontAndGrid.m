function fontAndGrid(h)
    grid on;
    FS = findall(h,'-property','FontWeight');
    FN = findall(h,'-property','FontName');
    set(FS,'FontWeight','bold');
    set(FN, 'FontName', 'Helvetica');
end