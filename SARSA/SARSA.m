% SARSA

% In the following we like to implement the SARSA algorithm to solve an
% MDP. The main difference w.r.t. Q-learning is the update rule:
%
% Q(s,a) = Q(s,a) + alpha * (inst_rew + gamma * Q(s_prime, a_prime) - Q(s,a))
%
% where s_prime and a_prime are the next state reached and played action.
% In a sens in Q-learning we update before doing the next action, while in
% SARSA we update the Q function only after we decided the action to
% perform.

clear
clc
close all

n_states = 3;
n_actions = 3;
gamma = 0.95;

allowed_actions = [1 1 0; 1 0 1; 1 0 0];

s = randi(3);
Q = zeros(n_states,n_actions);
M = 1000000;
m = 1;

policy = @eps_greedy;

a = policy(s, allowed_actions, Q, 1);

while m < M
    alpha = 1 - m/M;
    eps = sqrt(1 - m/M);

    % Environment
    [s_prime, inst_rew] = transition_model(s, a);

    % Eps-greedy
    a_prime = policy(s_prime, allowed_actions, Q, eps);

    % SARSA update
    Q(s,a) = Q(s,a) + alpha * (inst_rew + gamma * Q(s_prime, a_prime) - Q(s,a));
    s = s_prime;
    a = a_prime;
    m = m + 1;

end

Q
