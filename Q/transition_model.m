function [s_prime, inst_rew] = transition_model(s, a)

if s == 1
    if a == 1
        prob = [0.9 0.1 0];
    elseif a == 2
        prob = [0.3 0.7 0];
    end
end

if s == 2
    if a == 1
        prob = [0.4 0.6 0];
    elseif a == 3 
        prob = [0 0.3 0.7];
    end
end

if s == 3 && a == 1
    prob = [0.2 0 0.8];
end

cumprob = cumsum(prob);
s_prime = find(cumprob >= rand(),1);

% State 1
if s == 1 && a == 1 && s_prime == 1
    inst_rew = 0;
end
if s == 1 && a == 1 && s_prime == 2
    inst_rew = 20;
end
if s == 1 && a == 2 && s_prime == 1
    inst_rew = -2;
end
if s == 1 && a == 2 && s_prime == 2
    inst_rew = -27;
end

% State 2
if s == 2 && a == 1 && s_prime == 1
    inst_rew = 0;
end
if s == 2 && a == 1 && s_prime == 2
    inst_rew = 20;
end
if s == 2 && a == 3 && s_prime == 2
    inst_rew = -5;
end
if s == 2 && a == 3 && s_prime == 3
    inst_rew = -100;
end

if s == 3 && a == 1 && s_prime == 1
    inst_rew = 0;
end
if s == 3 && a == 1 && s_prime == 3
    inst_rew = 50;
end