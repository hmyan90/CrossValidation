function matX = transX(x)
   lens = size(x, 2);
   matX = ones(1, lens);
   matX = [matX; x; x.^2; x.^3; x.^4; x.^5; x.^6; x.^7; x.^8; x.^9];
endfunction
