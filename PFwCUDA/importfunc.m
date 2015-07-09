clear a; clear flipped; clear mapa; clear wallsa;
a = importdata('postcancel');

mapa = imread('acoustic_duct.bmp');

[widtha,heighta] = size(mapa);

wallsa = zeros(widtha, heighta);

% for y = 1:heighta,
%    for x = 1:widtha,
%        if mapa(x,y) == 0,
%           wallsa(x,y) = 1; 
%        end
%    end
% end
%a = a(1:523,1:858);
a = 10* log10(abs(a + 0.0000000000001));
a = a + 100;
flipped = flipud(a);
for y = 1:heighta,
   for x = 1:widtha,
      if (wallsa(x,y) == 1)
        flipped(x,y) = 0;
      end
   end
end
figure; image(flipped); axis off;

d = importdata('precancel')-importdata('postcancel');
d = 10*log10(abs(d + 0.00000000000001));
d = d + 200;
flipped = flipud(d);
for y = 1:heighta,
   for x = 1:widtha,
      if (wallsa(x,y) == 1)
        flipped(x,y) = 0;
      end
   end
end
figure; image(flipped); axis off;




