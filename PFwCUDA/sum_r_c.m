function [r,c] = sum_r_c(x)
    [width,height] = size(x);
    c = zeros(1, width);
    r = zeros(1, height);
   
    for i=1:width,
       c(i) = 0;
       for j = 1:height,
          if x(i,j) == 0,
             c(i)=c(i) + 1; 
          end
       end
    end

    for i=1:height,
       r(i) = 0;
       for j = 1:width,
          if x(j,i) == 0,
             r(i)=r(i) + 1; 
          end
       end
    end

end