% Q-Learning

% Now we like to analyse problems in which we are not provided complete
% information for the MDP we like to anaylise. In particular, we would like
% to learn the optimal policy to follow by relying on the information
% coming from the samples given by the MDP. We start by implementing the
% common Reinforcement Learning (RL) algorithm Q-Learning.

% We consider again the MDP modeling an advertising problem.

% If in the case of Dynamic Programming (DP) we were provided the model of
% the MDP in this case we are able to resort only on the transition model
% functions:
%
% r: S x A -> R
% P: S -> S
% 
% which can tell you the new state and the instantaneous reward if we
% provide a current state s and a chosen action a. So the analysis we might
% perform will only based on the generation of some episodes coming from
% the MDP.  Notice that in this case we do not have the expected
% instantaneous reward, but only a realization of the instantaneous reward.
%
% Moreover, we should consider a specific policy to run on our problem in
% order to select the next action. In this case we rely on the eps-greedy
% one: 

% eps_greedy(s, allowed_actions, Q, eps).
% 
% where we need to specify the current state s, the actions which can be
% chosen in the current state allowed_actions, the values we have for the
% Q function Q (here we assume to pass the complete matrix of state-action
% value function) and the value of epsilon (exploration rate) eps. 
% We will see how this parameter has to be modified during the learning
% process.

% Let us assume to have only the possibility to extract samples from the
% MDP. To run the Q-learning algorithm we need to specify:
%
% - The discount factor we are considering gamma y
% - The set of the allowed actions in each state
% - The initial state s_0
% - The initial Q function, usuall Q_0(s,a) = 0, FOR ALL s and a
% - The maximum number of iterations M
% - A policy to consider

n_states = 3;
n_actions = 3;
gamma = 0.95;

allowed_actions = [1 1 0; 1 0 1; 1 0 0];

s = randi(3);
Q = zeros(n_states, n_actions);
M = 500000;
m = 1;

policy = @eps_greedy;

% Once we initialize the algorithm, at each step we need to update the
% value of Q(s,a) by using the instantious reward we got r(s,a) with the
% following formula.
%
%
%
% where alpha is the learning rae and s' is the next state we reached. In
% principle during the learning process one should decrease both the
% learning rate and the exploration term, since otherwise we are not able
% to converge to the optimal policy. 

while m < M
    alpha = 1 - m/M;
    eps = (1-m/M)^2;
    
    % Eps-greedy
    a = policy(s, allowed_actions, Q, eps);
    
    % Environment 
    [s_prime, inst_rew] = transition_model(s,a);
    
    % Q-learing upsdate
    Q(s,a) = Q(s,a) + alpha * (inst_rew + gamma * max(Q(s_prime,:)) - Q(s,a));
    s = s_prime;
    m = m + 1;
end

Q

% Since it is an offline learning method, even if we do not follow the
% optimal policy at the end of the learning period, we still learn the
% optimal policy.