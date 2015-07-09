dl = 0.3;
sp = 340;
Fs = sp/dl;
T = 1/Fs;
signal = probes(:,2);
L = length(signal);
t = (0:L-1)*T;

NFFT = 2^nextpow2(L);
Y = fft(signal, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);

figure;
%hold on;
plot(f, 2*abs(Y(1:NFFT/2+1)),'r');


