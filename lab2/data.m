clc; clear
low = [1.0785 0.9870 1.1485 1.0934 1.0970 1.0031 1.1900 1.1337 0.9378...
       0.9759 0.9702];
   
medium = [0.9703 1.0049 0.9794 1.0241 1.2885 1.3803 1.1004 1.0727 1.4285...
          1.1442 1.1780 0.9666 1.0271 1.0006 1.0568 1.0751];
      
high = [1.0813 1.1416 1.4442 1.3347 1.3329 1.4435 1.1555 1.4146 1.3686...
        1.2521 1.1900 1.0663 1.0379 1.1450 1.3451 1.2727 1.8145 0.9799...
        1.2091 1.1952 1.3373 1.1462 2.7891 1.2878 1.0168 1.1723 1.8525];

figure()
hold on
edges=0.9:0.1:2.8;
h1 = histcounts(low, edges)/length(low);
h2 = histcounts(medium, edges)/length(medium);
h3 = histcounts(high, edges)/length(high);
b = bar(edges(1:end-1),[h1; h2; h3]',1);
b(1).FaceColor = [0.4660, 0.6740, 0.1880]
b(2).FaceColor = [0.9290, 0.6940, 0.1250]
b(3).FaceColor = [0.6350, 0.0780, 0.1840]
grid on
grid minor
xticks([0.9:0.2:2.8])
xlabel("Ratio")
ylabel("Frequency (%)")
legend('Low risk', 'Medium risk', 'Melanoma')
ylim([0 0.5])