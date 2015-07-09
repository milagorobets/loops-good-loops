clear all; close all;
map = imread('squareroom.bmp');
image(map);

fine_span = 24;

[width,height] = size(map);

col_sum = zeros(1, width);
row_sum = zeros(1, height);
map_r = zeros(width, height);

[col_sum, row_sum] = sum_r_c(map);

figure; plot(col_sum); xlabel('Column');
figure; plot(row_sum); xlabel('Row');

col_sum = col_sum - fine_span;
row_sum = row_sum - fine_span;

close all;
figure; plot(col_sum); xlabel('Column');
figure; plot(row_sum); xlabel('Row');


%identify starters for column boxes
%big boxes first

max_col = max(col_sum);
group_cut = 0.8*max_col;

pos = 0;

% for i = 1:width,
%    if col_sum(i) > group_cut,
%        % map(:,i) = 70;
%        pos = pos + 1;
%        vb_st(pos) = i;
%        
%    end
% end

% group ranges together
% for i = 2:pos,
%    if (vb_st(i) == (vb_st(i-1)+1))
%       vb_st(i) =  
%    end
% end


%different attempt to group
for x = 1:width,
    vl = 0;
    for y = 1:height,
        if (map(x,y) == 0) % wall pixel
            if (map(x,y-1) == 0)
               % add to range
               vb_sp(x,vl) = vb_sp(x,vl) + 1;
            else
                % start new range
                vl = vl + 1;
                vb_st(x,vl) = y;
                vb_sp(x,vl) = 1;
               
            end
            
        end
    end
end

for y = 1:height,
   hl = 0;
   for x = 1:width,
       if (map(x,y) == 0)
        if (map(x-1,y) == 0)
           hb_sp(hl,y) = hb_sp(hl,y) + 1; 
        else
            hl = hl + 1;
            hb_st(hl,y) = x;
            hb_sp(hl,y) = 1;
        end
       end
   end
end

image(map);


for y = 1: height,
   for x = 1:width,
        
   end
end