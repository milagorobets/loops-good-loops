%read in map
clear all; close all;

iter = 500;
dec = 0.5;
walldec = 0.1;
ampl = 0.4;

SAMPLES_TO_AVG = 100;

SPEED = 3e8;
CT = SPEED/2; % should be sqrt(2), but this is easier
DL = 2;
SRC_FREQ = 15e6;

map = imread('matlabenv.bmp');
image(map);

[width,height] = size(map);

outputmap = zeros(width,height);

inc_east = zeros(width,height);
inc_west = zeros(width,height);
inc_south = zeros(width,height);
inc_north = zeros(width,height);

srcloc = zeros(width,height);
srcloc(floor(width/2), floor(height/2)) = 1;

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

tic;
% compute all iter
for i = 1:iter,
    % update src
    src = ampl*sin(pi*i*DL*SRC_FREQ/CT);
    % compute
    for y = 1:height,
       for x = 1:width,
          if (y > 1) 
          sc_south(x,y)= dec * (inc_east(x,y-1) ...
                                + inc_north(x,y-1) ...
                                + inc_west(x,y-1) ...
                                - inc_south(x,y-1));             
          end
          
          if (x > 1)
          sc_east(x,y) = dec * (- inc_east(x-1,y) ...
                                + inc_north(x-1,y) ...
                                + inc_west(x-1,y) ...
                                + inc_south(x-1,y));
          end
          
          if (x < width)
          sc_west(x,y) = dec * (inc_east(x+1,y) ...
                                + inc_north(x+1,y)...
                                - inc_west(x+1,y) ...
                                + inc_south(x+1,y));
          end
          
          if (y < height)
          sc_north(x,y) = dec * (inc_east(x,y+1)...
                                - inc_north(x,y+1) ...
                                + inc_west(x,y+1) ...
                                + inc_south(x,y+1));
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
               inc_east(x,y) = src + t0;
               inc_west(x,y) = src + t1;
               inc_south(x,y) = src + t2;
               inc_north(x,y) = src + t3;
           else
                inc_east(x,y) = t1;
                inc_west(x,y) = t0;
                inc_south(x,y) = t3;
                inc_north(x,y) = t2;
           end
       end 
    end
    
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
% convert to db map
toc;

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
