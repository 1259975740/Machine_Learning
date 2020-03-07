mu = @(x) -1.9+.23*x;   
x = 5:.1:15;
yhat = mu(x);
dy = -3.5:.1:3.5; sz = size(dy); k = (length(dy)+1)/2;
x1 =  7*ones(sz); y1 = mu(x1)+dy; z1 = normpdf(y1,mu(x1),1);
x2 = 10*ones(sz); y2 = mu(x2)+dy; z2 = normpdf(y2,mu(x2),1);
x3 = 13*ones(sz); y3 = mu(x3)+dy; z3 = normpdf(y3,mu(x3),1);
subplot(1,2,1)
plot3(x,yhat,zeros(size(x)),'b-', ...
      x1,y1,z1,'r:', x1([k k]),y1([k k]),[0 z1(k)],'r-', ...
      x2,y2,z2,'r:', x2([k k]),y2([k k]),[0 z2(k)],'r-', ...
      x3,y3,z3,'r:', x3([k k]),y3([k k]),[0 z3(k)],'r-');
zlim([0 1]);
xlabel('x'); ylabel('Ey'); zlabel('条件概率密度');
grid on; view([-45 45]);


mu = @(x) exp(-1.9+.23*x);
x = 5:.1:15;
yhat = mu(x);
x1 =  7*ones(1,5);  y1 = 0:4; z1 = poisspdf(y1,mu(x1));
x2 = 10*ones(1,7); y2 = 0:6; z2 = poisspdf(y2,mu(x2));
x3 = 13*ones(1,9); y3 = 0:8; z3 = poisspdf(y3,mu(x3));
subplot(1,2,2)
plot3(x,yhat,zeros(size(x)),'b-', ...
      [x1; x1],[y1; y1],[z1; zeros(size(y1))],'r-', x1,y1,z1,'r.', ...
      [x2; x2],[y2; y2],[z2; zeros(size(y2))],'r-', x2,y2,z2,'r.', ...
      [x3; x3],[y3; y3],[z3; zeros(size(y3))],'r-', x3,y3,z3,'r.');
zlim([0 1]);
xlabel('X'); ylabel('Y'); zlabel('Probability');
grid on; view([-45 45]);