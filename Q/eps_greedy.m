function a = eps_greedy(s, allowed_actions, Q, eps)

if rand() <= eps
    actions = find(allowed_actions(s,:) == 1);
    a = randsample(actions,1);
else
    curr_Q = Q(s,:);
    curr_Q(~allowed_actions(s,:)) = -inf;
    [~, ind_action] = max(curr_Q);
    if length(ind_action) > 1
        a = randsample(ind_action,1);
    else
        a = ind_action;
    end
end