function H =  bs_compute_H(A, D, X, Y, V, para)

A = A.*D;
    H = bs_compute_H1(single(A), int32(X)-1, int32(Y)-1, single(V));
%     toc;
H = H/2;