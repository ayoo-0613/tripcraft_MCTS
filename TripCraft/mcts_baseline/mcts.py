from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _puct_select(node: _Node, c: float = 1.4) -> _Node:
    assert node.children
    log_n = math.log(node.visits + 1)

    def score(ch: _Node) -> float:
        if ch.visits == 0:
            return float("inf")
        prior = ch.prior if ch.prior > 0 else 1.0
        return ch.value + c * prior * math.sqrt(log_n / ch.visits)

    return max(node.children, key=score)


def _best_path_actions(root: _Node) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    node = root
    while node.children:
        node = max(node.children, key=lambda ch: (ch.visits, ch.value))
        if node.action:
            actions.append(node.action)
    return actions


def mcts_search(
    env: TripCraftEnv,
    rollouts: int,
    topk: int,
    *,
    guidance: Optional[GuidanceModel] = None,
    prior_c: float = 1.4,
    value_weight: float = 0.0,
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

    for _ in range(int(rollouts)):
        node = root

        # Selection
        while not env.is_terminal(node.state) and node.untried_actions == [] and node.children:
            node = _puct_select(node, c=prior_c)

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
                child = _Node(state=next_state, parent=node, action=action, prior=prior)
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
