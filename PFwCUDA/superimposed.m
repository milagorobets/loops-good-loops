testx = 0:0.001:1;
sin1 = sin(2*pi*testx./0.1);
figure; plot(sin1);
esin1 = sin1.*exp(testx/10);
figure; plot(esin1);

esin2 = fliplr(esin1);
figure; plot(esin1, 'b'); hold on; plot(esin2,'r');
eesum = esin1 + esin2;
plot(eesum, 'g');
