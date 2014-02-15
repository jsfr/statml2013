function setPaper(h) 
    W = 4; H = 3;
    set(h,'PaperUnits','inches');
    set(h,'PaperOrientation','portrait');
    set(h,'PaperSize',[H,W]);
    set(h,'PaperPosition',[0,0,W,H]);
end