from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .env import State, TripCraftEnv


@dataclass
class _Node:
    state: State
    parent: Optional["_Node"] = None
    action: Optional[Dict[str, Any]] = None
    children: List["_Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    untried_actions: Optional[List[Dict[str, Any]]] = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _uct_select(node: _Node, c: float = 1.4) -> _Node:
    assert node.children
    log_n = math.log(node.visits + 1)

    def score(ch: _Node) -> float:
        if ch.visits == 0:
            return float("inf")
        return ch.value + c * math.sqrt(log_n / ch.visits)

    return max(node.children, key=score)


def _best_path_actions(root: _Node) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    node = root
    while node.children:
        node = max(node.children, key=lambda ch: (ch.visits, ch.value))
        if node.action:
            actions.append(node.action)
    return actions


def mcts_search(env: TripCraftEnv, rollouts: int, topk: int) -> State:
    """
    Run MCTS from initial state to terminal state.
    Expansion uses env.legal_actions(state, topk=topk).
    Rollout uses greedy completion with same retrieval topk.
    Return best terminal state by reward.
    """
    rng = random.Random(0)
    root_state = env.initial_state()
    root = _Node(state=env.clone_state(root_state))

    for _ in range(int(rollouts)):
        node = root

        # Selection
        while not env.is_terminal(node.state) and node.untried_actions == [] and node.children:
            node = _uct_select(node)

        # Expansion
        if not env.is_terminal(node.state):
            if node.untried_actions is None:
                node.untried_actions = env.legal_actions(node.state, topk=topk)
            if node.untried_actions:
                action = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
                next_state = env.clone_state(node.state)
                next_state = env.apply_action(next_state, action, record_trace=False)
                child = _Node(state=next_state, parent=node, action=action)
                node.children.append(child)
                node = child
            else:
                node.untried_actions = []

        # Simulation
        terminal = env.greedy_rollout(node.state, topk=topk, record_trace=False)
        reward = env.evaluate(terminal)

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
