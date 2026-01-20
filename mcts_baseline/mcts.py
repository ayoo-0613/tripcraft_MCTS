from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .env import State, TripCraftEnv
from .guidance import GuidanceModel, NullGuidance


@dataclass
class _Node:
    state: State
    parent: Optional["_Node"] = None
    action: Optional[Dict[str, Any]] = None
    children: List["_Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    untried_actions: Optional[List[Dict[str, Any]]] = None
    prior_map: Dict[int, float] = field(default_factory=dict)
    prior: float = 1.0
    persona_bonus: float = 0.0
    risk: float = 0.0
    slack_factor: float = 1.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _puct_select(
    node: _Node,
    c: float = 1.4,
    persona_weight: float = 0.0,
    risk_weight: float = 0.0,
    use_slack: bool = False,
) -> _Node:
    assert node.children
    log_n = math.log(node.visits + 1)

    def score(ch: _Node) -> float:
        if ch.visits == 0:
            return float("inf")
        prior = ch.prior if ch.prior > 0 else 1.0
        if use_slack:
            prior *= ch.slack_factor
        return (
            ch.value
            + c * prior * math.sqrt(log_n / ch.visits)
            + persona_weight * ch.persona_bonus
            - risk_weight * ch.risk
        )

    return max(node.children, key=score)


def _best_path_actions(root: _Node) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    node = root
    while node.children:
        node = max(node.children, key=lambda ch: (ch.visits, ch.value))
        if node.action:
            actions.append(node.action)
    return actions


def _coerce_violation_count(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return float(sum(1 for v in value.values() if v))
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        head = value[0]
        if isinstance(head, (bool, int, float)):
            return float(head)
        return float(sum(1 for v in value if v))
    return 0.0


def _hard_violation_count(
    env: TripCraftEnv,
    state: State,
    hard_violation_fn: Optional[Callable[[TripCraftEnv, State], Any]],
) -> float:
    if hard_violation_fn is not None:
        return _coerce_violation_count(hard_violation_fn(env, state))
    for name in ("hard_violation_count", "hard_constraint_violation", "hard_violation"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return _coerce_violation_count(fn(state))
            except Exception:
                continue
    return 0.0


def _resource_slack(
    env: TripCraftEnv,
    state: State,
    action: Dict[str, Any],
    slack_fn: Optional[Callable[[TripCraftEnv, State, Dict[str, Any]], float]],
    slack_floor: float,
) -> float:
    if slack_fn is not None:
        try:
            return float(slack_fn(env, state, action))
        except Exception:
            return 1.0
    fn = getattr(env, "resource_slack", None)
    if callable(fn):
        try:
            return float(fn(state, action))
        except Exception:
            try:
                return float(fn(state))
            except Exception:
                pass
    budget = float(getattr(getattr(env, "row", None), "budget", 0.0) or 0.0)
    if budget <= 0.0:
        return 1.0
    cost = 0.0
    if hasattr(env, "_action_cost"):
        try:
            cost = float(env._action_cost(state, action))
        except Exception:
            cost = 0.0
    remaining = budget - (state.budget_used + cost)
    slack = remaining / budget
    if slack_floor > 0.0:
        slack = max(slack, slack_floor)
    return max(0.0, min(slack, 1.0))


def _advance_action_for_lookahead(env: TripCraftEnv, state: State, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for action in actions:
        if action.get("type", "").startswith("skip_"):
            return action
    if hasattr(env, "_budget_ok"):
        for action in actions:
            try:
                if env._budget_ok(state, action):
                    return action
            except Exception:
                continue
    return actions[0] if actions else None


def _lookahead_risk(
    env: TripCraftEnv,
    state: State,
    *,
    topk: int,
    horizon: int,
    saturate_k: int,
) -> float:
    if horizon <= 0:
        return 0.0
    s = env.clone_state(state)
    counts: List[int] = []
    for _ in range(horizon):
        if env.is_terminal(s):
            break
        actions = env.legal_actions(s, topk=topk)
        counts.append(len(actions))
        action = _advance_action_for_lookahead(env, s, actions)
        if action is None:
            break
        before = (s.day, s.substep, s.budget_used, s.done)
        s = env.apply_action(s, action, record_trace=False)
        after = (s.day, s.substep, s.budget_used, s.done)
        if after == before:
            break
    if not counts:
        return 0.0
    norm_k = max(1, saturate_k)
    comp = sum(min(c / norm_k, 1.0) for c in counts) / float(len(counts))
    return 1.0 - comp


def mcts_search(
    env: TripCraftEnv,
    rollouts: int,
    topk: int,
    *,
    guidance: Optional[GuidanceModel] = None,
    prior_c: float = 1.4,
    value_weight: float = 0.0,
    enable_risk_penalty: bool = False,
    risk_lambda: float = 0.0,
    risk_horizon: int = 2,
    risk_saturate_k: Optional[int] = None,
    enable_slack_modulation: bool = False,
    slack_floor: float = 0.0,
    slack_fn: Optional[Callable[[TripCraftEnv, State, Dict[str, Any]], float]] = None,
    enable_feasibility_gate: bool = False,
    feasibility_weight: float = 1.0,
    hard_violation_fn: Optional[Callable[[TripCraftEnv, State], Any]] = None,
) -> State:
    """
    Run MCTS from initial state to terminal state.
    Expansion uses env.legal_actions(state, topk=topk).
    Rollout uses greedy completion with same retrieval topk.
    Return best terminal state by reward.
    """
    rng = random.Random(0)
    guide = guidance or NullGuidance()
    root_state = env.initial_state()
    root = _Node(state=env.clone_state(root_state))
    risk_on = enable_risk_penalty and risk_lambda > 0.0
    slack_on = enable_slack_modulation
    gate_on = enable_feasibility_gate
    if risk_saturate_k is None or risk_saturate_k <= 0:
        risk_saturate_k = max(1, topk)
    slack_floor = max(0.0, min(slack_floor, 1.0))

    for _ in range(int(rollouts)):
        node = root

        # Selection
        while not env.is_terminal(node.state) and node.untried_actions == [] and node.children:
            node = _puct_select(
                node,
                c=prior_c,
                persona_weight=getattr(env, "PERSONA_UCT_WEIGHT", 0.0),
                risk_weight=risk_lambda if risk_on else 0.0,
                use_slack=slack_on,
            )

        # Expansion
        if not env.is_terminal(node.state):
            if node.untried_actions is None:
                actions = env.legal_actions(node.state, topk=topk)
                priors = guide.prior(node.state, actions, env)
                if not priors or len(priors) != len(actions):
                    priors = [1.0 / len(actions)] * len(actions) if actions else []
                ranked = sorted(zip(actions, priors), key=lambda ap: ap[1], reverse=True)
                node.untried_actions = [a for a, _ in ranked]
                node.prior_map = {id(a): p for a, p in ranked}
            if node.untried_actions:
                action = node.untried_actions.pop(0)
                next_state = env.clone_state(node.state)
                next_state = env.apply_action(next_state, action, record_trace=False)
                prior = node.prior_map.get(id(action), 1.0)
                persona_bonus = env.persona_action_score(action) if hasattr(env, "persona_action_score") else 0.0
                slack_factor = 1.0
                if slack_on:
                    slack_factor = _resource_slack(env, node.state, action, slack_fn, slack_floor)
                risk = 0.0
                if risk_on:
                    risk = _lookahead_risk(
                        env,
                        next_state,
                        topk=topk,
                        horizon=risk_horizon,
                        saturate_k=risk_saturate_k,
                    )
                child = _Node(
                    state=next_state,
                    parent=node,
                    action=action,
                    prior=prior,
                    persona_bonus=persona_bonus,
                    risk=risk,
                    slack_factor=slack_factor,
                )
                node.children.append(child)
                node = child
            else:
                node.untried_actions = []

        # Simulation
        terminal = env.greedy_rollout(node.state, topk=topk, record_trace=False)
        reward = env.evaluate(terminal)
        if value_weight > 0.0:
            v = guide.value(node.state, env)
            reward = (1.0 - value_weight) * reward + value_weight * v
        if gate_on:
            violations = _hard_violation_count(env, terminal, hard_violation_fn)
            if violations > 0.0:
                reward = -abs(feasibility_weight) * violations

        # Backpropagation
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent

    # Reconstruct a single action sequence and replay to get a clean terminal state.
    actions = _best_path_actions(root)
    replay = env.initial_state()
    for a in actions:
        replay = env.apply_action(replay, a, record_trace=True)
        if env.is_terminal(replay):
            break
    return env.greedy_rollout(replay, topk=topk, record_trace=True)
