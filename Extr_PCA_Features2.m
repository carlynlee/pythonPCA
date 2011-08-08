function [phi projMatA]=Extr_PCA_Features2(xraw,indInt)

% indInt=TrainIndclass2;
ns=length(indInt);
    
x=xraw(:,indInt);
% xmean=mean(x,2);
% 
% % Remove means
% for i=1:size(x,2)
%     x(:,i)=x(:,i)-xmean;
% end



gridSize = size(x,1);
theta = zeros(ns);
for i=1:ns
    vi = x(:,i);
    for j=i:ns
        vj=x(:,j);
        theta(i,j) = vi'*vj/gridSize/ns;
        theta(j,i) = theta(i,j);
    end
end

% Compute dominant eigenvalue/eigenvector
% [u,lam] = eig(theta);
% lam=diag(lam);
% [lam, ind] = sort(lam,'descend');
% u=u(:,ind');
[u,lam] = eig(theta);
[lam, ind] = sort(-diag(lam));
lam=-lam;
u=u(:,ind');
% Normalize so that first eigenvector has unit length
for i=1:ns
    u(:,i)=u(:,i)/sum(abs(u(:,i)));
end

% Normalized eigenvectors...
for i=1:ns
    phi(:,i)=zeros(gridSize,1);
    for j=1:ns
        phi(:,i)=phi(:,i)+u(j,i)*x(:,j);
    end
    phi(:,i)=phi(:,i)/norm(phi(:,i));
    normPhi(i)=phi(:,i)'*phi(:,i);
end

% Determine sign for dominant eigenvector...select sign so that the sum
% of the projections of the tumor samples used for the POD is greater
% than 1.
clear projMatA
for j=1:ns
    sumP1(j)=0.;
    for i=1:ns
        sumP1(j)=sumP1(j)+sum(x(:,i)'*phi(:,j)/normPhi(j));
    end
    if sumP1(j) < 0, phi(:,j)=-phi(:,j); end
    
    % Projections...
    for i=1:size(xraw,2)
        projMatA(i,j) = xraw(:,i)'*phi(:,j)/normPhi(j);
    end
end

