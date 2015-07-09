%read in map
clear all; close all;

pix_div = 1;
%iter = floor(100/pix_div);
iter = 5000;

dec = 0.5;
walldec = 0.001^pix_div;
ampl = 0.1;

SAMPLES_TO_AVG = 100;

SPEED = 340;
CT = SPEED/2; % should be sqrt(2), but this is easier
DL = 0.1*pix_div;

SRC_FREQ = 17;

map = imread('p11m.bmp');
%image(map);

[width,height] = size(map);

outputmap = zeros(width,height);

inc_east = zeros(width,height);
inc_west = zeros(width,height);
inc_south = zeros(width,height);
inc_north = zeros(width,height);

srcloc = zeros(width,height);
%sx = floor(width/2); 
%sy = 128-25;
sy = 310; sx = 60;
%sy = floor(height/2);
srcloc(sx, sy) = 1;
%srcloc(3,3) = 1;

avg = zeros(width,height);

sc_east = zeros(width,height);
sc_west = zeros(width,height);
sc_south = zeros(width,height);
sc_north = zeros(width,height);

walls = zeros(width,height);

% find walls
for y = 1:height,
   for x = 1:width,
       if map(x,y) == 0,
          walls(x,y) = 1; 
       end
   end
end

lambda = SPEED/SRC_FREQ;
phs = DL/lambda;

tic;
% compute all iter
for i = 1:iter,
    % update src
    src = ampl*sin(pi*i*DL*SRC_FREQ/CT); % * pix_div maybe to account for DL
    % compute
    for y = 1:pix_div:height,
       for x = 1:pix_div:width,
          scs = 0; sce = 0; scw = 0; scn = 0; 
          if (y > pix_div) 
          scs = dec * (inc_east(x,y-pix_div) ...
                                + inc_north(x,y-pix_div) ...
                                + inc_west(x,y-pix_div) ...
                                - inc_south(x,y-pix_div));             
          end
          
          if (x > pix_div)
          sce = dec * (- inc_east(x-pix_div,y) ...
                                + inc_north(x-pix_div,y) ...
                                + inc_west(x-pix_div,y) ...
                                + inc_south(x-pix_div,y));
          end
          
          if (x < width-pix_div)
          scw = dec * (inc_east(x+pix_div,y) ...
                                + inc_north(x+pix_div,y)...
                                - inc_west(x+pix_div,y) ...
                                + inc_south(x+pix_div,y));
          end
          
          if (y < height-pix_div)
          scn = dec * (inc_east(x,y+pix_div)...
                                - inc_north(x,y+pix_div) ...
                                + inc_west(x,y+pix_div) ...
                                + inc_south(x,y+pix_div));
          end
          
          % populate neighbouring ones
          for k = 0:1:(pix_div-1),
             for m = 0:1:(pix_div-1),
                sc_south(x+m, y+k) = scs*1^(m+k); %try phase shift
                sc_east(x+m, y+k) = sce*1^(m+k);
                sc_west(x+m, y+k) = scw*1^(m+k);
                sc_north(x+m, y+k) = scn*1^(m+k);
             end
          end
       end
    end
    % copy nm to m
    for y = 1:height,
       for x = 1:width, 
           
           t0 = sc_east(x,y);
           t1 = sc_west(x,y);
           t2 = sc_south(x,y);
           t3 = sc_north(x,y);
           
           if ((x == 1) && (t0 == 0))
                t0 = sc_east(x+1,y);
           end
           
           if ((x == width) && (t1 == 0))
                t1 = sc_west(x-1,y);
           end
           
           if ((y == 1) && (t2 == 0))
                t2 = sc_south(x,y+1);
           end
           
           if ((y == height) && (t3 == 0))
                t3 = sc_north(x,y-1);
           end
           
           if (walls(x,y) == 1)
               inc_east(x,y) = walldec*t0;
               inc_west(x,y) = walldec*t1;
               inc_south(x,y) = walldec*t2;
               inc_north(x,y) = walldec*t3;
           elseif(srcloc(x,y) == 1)
               
               % populate neighbouring ones
                for k = 0:1:(pix_div-1),
                    for m = 0:1:(pix_div-1),
                        inc_east(x+m,y+k) = src + t0;
                        inc_west(x+m,y+k) = src + t1;
                       inc_south(x+m,y+k) = src + t2;
                       inc_north(x+m,y+k) = src + t3;
                     
                     end
                  end
           else
                inc_east(x,y) = t1;
                inc_west(x,y) = t0;
                inc_south(x,y) = t3;
                inc_north(x,y) = t2;
           end
       end 
    end
    
%     for k = 0:1:(pix_div-1),
%                     for m = 0:1:(pix_div-1),
%                         inc_east(sx+m,sy+k) = src + t0;
%                         inc_west(sx+m,sy+k) = src + t1;
%                        inc_south(sx+m,sy+k) = src + t2;
%                        inc_north(sx+m,sy+k) = src + t3;
%                      
%                      end
%                   end
    
    % add and average
    for y = 1:height,
       for x = 1:width,
           total = inc_east(x,y) + inc_west(x,y) + ...
                    inc_north(x,y) + inc_south(x,y);
%            total = total * 0.5; % ??
           total = total * total;
           oldavg = avg(x,y);
           total = oldavg*(SAMPLES_TO_AVG - 1) + total;
           avg(x,y) = total/SAMPLES_TO_AVG;
       end
    end
end
toc;

% convert to db map

for y = 1:height,
   for x = 1:width,
      level = avg(x,y);
      level = 10 * log10(abs(level + 0.0000000000001));
      
      outputmap(x,y) = level + 100;
%       if (level < -100) %red
%          outputmap(x,y) = 75; 
%       elseif (level < -90) %blue
%          outputmap(x,y) = 15;
%       elseif (level < -80) %orange
%          outputmap(x,y) = 'cyan'; % CHANGE THIS
%       elseif (level < -70) % yellow
%          outputmap(x,y) = 'yellow';
%       else %green
%          outputmap(x,y) = 'green';
%       end  
      
      if (walls(x,y) == 1)
         outputmap(x,y) = 0; 
      end
   end
end

% testjune19 = testjune19(1:425,1:1019);
% testjune19 = 10* log10(abs(testjune19 + 0.0000000000001));
% testjune19 = testjune19 + 100;
% flipped = flipud(testjune19);
% for y = 1:height,
%    for x = 1:width,
%       if (walls(x,y) == 1)
%         flipped(x,y) = 0;
%       end
%    end
% end

% display
figure; contourf(outputmap);
figure; image(outputmap);
