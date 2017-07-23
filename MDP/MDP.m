% Markov Decision Processes 

% We want to analyse the existing methods to evaluate the performance of a
% given policy on a Markov Decision Process (MDP). Subsequently, we will
% apply the ones able to find the optimal policy in an MDP.

% 1. Computing values on a MDP

% Let us consider a MDP which models an advertising problem. We would like
% to model when we have fixed policy. We just need to define the transition
% probabilities matrix and the immediate expected return for each state.

clear
clc
close all

n_states = 3;
gamma = 0.9;

R = [
 0.9*0 + 0.1*20;
 0.4*0 + 0.6*20;
 0.2*0 + 0.8*50;
];

P = [
    0.9 0.1 0; 
    0.4 0.6 0; 
    0.2 0 0.8
];

% Since we fixed the policy, we reduced the MDP to a Markov Process. At
% this point we are interested in the computation of the value for each
% state of the MDP. This could be performed with different tools. For
% instance, one might resort to the Bellman expecation equation.
%
% V = (I - yP)^-1 * R 
% 
% Since P is a stochastic matrix, we have some propertires on the
% eigenvalue of the matrix (I - y*P)

eig(P)
eig(gamma*P)
eig(eye(n_states) - gamma*P)

% Clearly we could not consider a discount factor gamma=1 otherwise the
% matrix would be singular and we would not be able to solve the Bellman
% equation.

% Thus the solution is:

V_eq = inv(eye(n_states) - gamma*P)* R

% This solution obtained by inverting the matrix is feasible only if the
% number of the states is finite and small. In fact, we require a
% computation cost of s^3 for the matrix inversion.

% Another posibility is to consider the recursive forumulation of the
% Bellman expecation equation and apply it iteratively.

V_old = zeros(n_states, 1);
tol = 0.0001;
V = R;

while any(abs(V_old - V) > tol) 
    V_old = V;
    V = R + gamma * P * V
end

[V V_eq]

% Here we stop when we are smaller than a determined tolerance level.

% The same procedure can be applied to the action-value function Q(s,a). In
% this case we explicitely enumerate the transitions P(s'|a,s) and the
% returns R(s,a) for each possible state-action pair.

R_sa = [
    0.9*0 + 0.1*20;
    0.3*-2 + 0.7*-27;
    0.4*0 + 0.6*20;
    0.3*-5 + 0.7*-100;
    0.2*0 + 0.8*50;
];

P_sas = [
    0.9 0.1 0;
    0.3 0.7 0;
    0.4 0.6 0;
    0 0.3 0.7;
    0.2 0 0.8;
];

policy = [
    1 0 0 0 0;
    0 0 1 0 0;
    0 0 0 0 1;
];

% In this case the values V and the transition probabilities P
% corresponding to the chosen policy are

policy * R_sa;
policy * P_sas;

% By resorting to the Bellman expectation equation in Q(s,a) we can solve
% the problem by means of a single matrix inversion

Q_eq = inv(eye(5) - gamma * P_sas * policy) * R_sa;

% Alternatively we can resort to the recursive solution provided by the
% Bellman expectation equation for Q:

Q = R_sa;
Q_old = zeros(5,1);

tol = 0.001;
n_rep = 0;

gamma = 0.9;
while any(abs(Q - Q_old) > tol)
    Q_old = Q;
    Q = R_sa + gamma * P_sas * policy * Q;
end

[Q Q_eq]

% Let us evaluate two different policies and compare them:
% 
% - far-sighted: we want to spend some money in marketing for the customer
% in both cases if she is a new customer or if she repeatedly purchased
% from our business
%
% - myopic: we do not want to spend any money in marketing
%
% The corresponding policy matrixes are

policy_far = [
    0 1 0 0 0;
    0 0 0 1 0;
    0 0 0 0 1;
];

policy_myo = [
    1 0 0 0 0;
    0 0 1 0 0;
    0 0 0 0 1;
];

% Let us evaluate the values of the different states by considering
% different discount factors y in the problem

gamma = 0.5
V_far = inv(eye(n_states) - gamma* policy_far * P_sas)* policy_far * R_sa
V_myo = inv(eye(n_states) - gamma* policy_myo * P_sas)* policy_myo * R_sa

gamma = 0.9
V_far = inv(eye(n_states) - gamma* policy_far * P_sas)* policy_far * R_sa
V_myo = inv(eye(n_states) - gamma* policy_myo * P_sas)* policy_myo * R_sa

gamma = 0.99
V_far = inv(eye(n_states) - gamma* policy_far * P_sas)* policy_far * R_sa
V_myo = inv(eye(n_states) - gamma* policy_myo * P_sas)* policy_myo * R_sa

% As you can see by setting different values for gamma it is possible to
% have different behavior for different policies. With gamma=0.5 the myopic
% policy outperforms the farsighted one, but as we increase gamme we have
% an incremental improvement of the farsighted one. Thus the consequent
% question is whether we are able to select the best policy for a given
% gamma.