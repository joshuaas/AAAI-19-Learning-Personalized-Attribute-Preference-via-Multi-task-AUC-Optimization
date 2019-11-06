function [W, model, funcVal] =  GLUBOOSTAUCPROX(X, Y, lambda1, lambda2,lambda3, nk, opts)
if nargin <7
    error('\n Inputs: X, Y, and lambda1,  lambda2, lambda2 should be specified!\n');
end
if nargin <8
    opts = [];
end

% initialize options.
opts=init_opts(opts);

% initial Lipschiz constant. 
if isfield(opts, 'lFlag')
    lFlag = opts.lFlag;
else
    lFlag = false;  
end

dimension = size(X{1}, 2);

num_u = numel(X) ;
% initialize a starting point
    P0 = zeros(dimension, num_u);
    U0 = zeros(dimension, num_u) ;
    theta0 = zeros(dimension, 1) ;



d = dimension ;
coef = 1;
% Set an array to save the objective value
funcVal = [];
%% initialization and precalculation
P = P0;
theta = theta0;
Up = U0;
Y = cellfun(@(elem) transY(elem),Y, 'UniformOutput', false ) ;
%  Y = Y';
XtX = cellfun(@(x,y)calXTLX(x,y) , X, Y, 'UniformOutput', false); 
XtY = cellfun(@(x,y) calXTLY(x,y), X, Y, 'UniformOutput', false);

n = min(dimension,num_u);

L = 1; 
% Initial function value
Pn = P; 
thetan = theta;
Upn = Up; 
funcVal = cat(1, funcVal, eval_loss());
%count = 0;
%  opts.maxIter = 80;
 t_new = 1;
W=  0 ;
% opts.maxIter  =100 ;
for iter = 1:opts.maxIter
    P_old = P;  theta_old = theta ; Up_old = Up;
%     Pn = P_old; thetan = theta_old ;
     t_old = t_new;
  
    grad_P = zeros(size(P));
    grad_theta = zeros(size(theta)) ;
    grad_Up  =zeros(size(Up)) ;
    for u = 1:num_u
      delta =  Pn(:, u)  + thetan + Upn(:, u);
      delta =   (coef)  * (XtX{u} * delta  - XtY{u}) ;

      grad_P(:, u)            = grad_P(:, u) + delta;
      grad_Up(:, u)         = grad_Up(:, u) + delta ;
      grad_theta            = grad_theta  + delta ;
    end
    % If we estimate the upper bound of Lipschitz constant, no line search
    % is needed.
    if lFlag
        update_param() ;
    else
        % line search 
        for inneriter = 1:100
            update_param() ;
            dP = P - Pn;    dtheta = theta - thetan ; dUp =Up -Upn ;  
            l00 = eval_auc_loss(thetan, Pn, Upn) ;
            Lhs = eval_diff();
%             Lhs1 = 0;

%             for c = 1:nc
%                 for u = 1:num_u
%                     dW = dP(:, u) + dtheta(:, c) ; 
%                     Lhs1 = Lhs1 +  (coef )  * 0.5 *  dW' * XtX{u} * dW ;
%                 end
%             end
%             
            Rhs = (L/2) * ( sumsqr(dP) + sumsqr(dtheta) + sumsqr(dUp) ) ; 
            if  Lhs <= Rhs
%                  update_param() ;
            	break ;
            else
                L = L*2;
            end
        end
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=10
                if (( funcVal(end-1) - funcVal(end) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=3
                if (( funcVal(end-1) -funcVal(end) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
        funcVal = cat(1, funcVal, eval_loss()); 

%    Update the coefficient
    t_new = ( 1 + sqrt( 1 + 4 * t_old^2 ) ) / 2;
    dt    = (t_old-1) / t_new ;
    % Update reference points 
     Pn    = P + dt * (P - P_old);
  
    Upn    = Up + dt * (Up - Up_old);
    
   thetan = theta + dt * (theta - theta_old) ;
%       Pn = P; Upn = Up; thetan = theta ;
end

W = theta * ones(1, num_u) + P  + Up;



model.W = W ;
model.P = P ;
model.theta =theta ;
model.Up = Up ;


     function res = transY(Y)
          res = Y ;
          if min(Y) == -1
              res = Y;
              res(res == -1) = 0;
          end
     end
     function Y =  l2norm(X) 
     	Y = sumsqr(X) ;
     end

     function Y = l12norm(X)
     	Y = sum(sqrt(sum(X.^2))) ;
     end
     
     function W =  proximalL12(X, thresh)
        W  = zeros(size(X)) ;
         for ncol =1:size(X,2)
            normtemp =  sqrt(sumsqr(X(:,ncol)))  ;
            W(:, ncol)=  max(normtemp - thresh, 0 ) * X(:, ncol) / (normtemp) ;
        end
     end
     function Y =  WNN(X, k)
          s= svd(X,'econ');
                  Y = sumsqr( s(  (k+1) : end) ) ;
     end

     function Y  = proximalL2norm(X,cons)
     	Y =   X / (1 + 2 * cons);
     end

     function Y =  proximalWNN(X ,k ,cons)
          [U, S, V]= svd(X,'econ');
          S = diag(S) ;
%           S((end-k+1):end)  = max( S((end-k+1):end) - cons, 0);
          S( (k+1) : end )  = S( (k+1) : end )/ (2 * cons + 1);
          S = diag(S) ;
          Y=  U * S * V' ;
     end

%      function Wsol =  proximalL12Sq(W,C,cons, iter)
%            loss_l12 = zeros(iter,1) ;  
% %            W = randn(size(C)) ;
%      		for  ii = 1:iter                 
%      			normW = (sum(abs(W), 2));
%      			weight = 1 ./ (abs(W) +eps) ;
%      			weight = (normW)  .* weight   ;
%      			weight = 1 ./ (2 * cons *weight + 1) ;
%      		    W = C  .*  weight ;
%                 loss_l12(ii) =  0.5 * sumsqr(W-C)  +cons *  l12normSq(W) ;  
%            end
%          Wsol = W; 
%      end
    function update_param()

        theta = proximalL2norm( thetan - grad_theta/L, lambda1/L);
        P = proximalWNN(Pn -  grad_P/L, nk,lambda3/L) ;
        Up    = proximalL12(Upn - grad_Up/L, lambda2/L) ;
    end 
    
    function res = calXTLX(A, B)
      np  = sum(B == 1) ;
      nn =  sum(B == 0) ;
      Xp  = A' * B / (np ) ;
      Xn = A' * (1-B) / (nn );
      D  = B /  np  + (1-B) / nn ;
      res = -Xp * Xn' - Xn * Xp';
      DX = bsxfun(@times, D, A) ;
      res = res + A' * DX ;
    end

    function res = calXTLY(X, Y)
      np  = sum(Y == 1) ;
      nn =  sum(Y == 0) ;
      Xp  = X' * Y / (np) ;
      Xn = X' * (1-Y)  / (nn);
      res = Xp  - Xn ;
    end

    function lu = eval_auc_loss(theta, P, Up)
         lu = 0;
          for un  = 1:num_u 
              np = sum(Y{un} == 1) ;
              nn =sum(Y{un} == 0 ) ;
              temp =  (Y{un} - X{un}  * (theta + P(:, un) + Up(:, un) ) ) ;
              lp  = temp' * Y{un} / np  ;
              ln = temp'  * (1 - Y{un}) /nn;
              llp = (temp .*  Y{un} /np)' * temp;
              lln = (temp .* (1- Y{un})  / nn )' * temp ;
              lu =  lu +  llp + lln - 2 * lp * ln ;
          end
        
        lu  = lu  *  0.5 * coef  ; 

    end
%   function lu = eval_auc_loss_org(theta, P)
%          lu = 0;
%         for cn = 1:nc 
%         	  for un  = 1:num_u 
%               tempY = Y{un} ;
%               tempX  = X{un} ;
%               ddelta =  theta(:, cn) + P(:, un) ;
%               np = sum(tempY == 1) ;
%               nn =sum(tempY == 0 ) ;
%               luc =  0 ;
%                 IndexP = tempY== 1;
%                 IndexN = tempY  == 0 ;
%                 predP = tempX(IndexP, :) *(ddelta ) ;
%                 predN = tempX(IndexN, :) *ddelta ;
%                     for ip = 1:size(predP,1)
%                         for in = 1:size(predN, 1)
%                             luc = luc +  (1- ( predP(ip) - predN(in) ) )^2;
%                         end
%                     end
%                 luc =  luc / (np * nn) ;
%                 lu = lu + luc ;
%         	  end
%         end
%         lu = lu * coef * 0.5 ;
%     end
    function l = eval_diff()
           l1 =eval_auc_loss(theta, P, Up) ;
           l = sum( sum( grad_P .* dP ) ) + ...
             	sum( sum( grad_theta .* dtheta ) ) + sum( sum( grad_Up .* dUp ) )  ; 
           l = l1-l00-l ;
           
    end

    function l = eval_loss()
        l = eval_auc_loss(theta, P,  Up) + lambda1 * l2norm(theta) + lambda2 * l12norm(Up) + lambda3 *WNN(P,nk) ;
  
     end
end