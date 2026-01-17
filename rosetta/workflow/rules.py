"""Action rules for workflow state transitions.

This module defines rules that constrain which actions are available
at each step of the workflow based on the current state and history.
"""

from typing import List, Optional, Callable

from rosetta.workflow.actions import StateResult


class ActionRule:
    """Collection of rule functions as static methods.

    Each rule takes (current_state, state_sequence, actions) and returns
    a filtered list of actions. Rules are composable via ActionRuleEnforcer.
    """

    @staticmethod
    def require_initial_plan_or_think(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """Initial state must start with plan or think."""
        if current_state == "initial":
            filtered = [a for a in actions if a in ("plan", "think")]
            return filtered if filtered else actions
        return actions

    @staticmethod
    def require_plan_before_execute(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """Can't execute without a prior plan."""
        if not state_sequence or not any(r.state == "plan" for r in state_sequence):
            return [a for a in actions if a not in ("execute", "parallel_execute")]
        return actions

    @staticmethod
    def require_continue_before_answer(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """Answer only appears after continue."""
        if not state_sequence or not any(r.state == "continue" for r in state_sequence):
            return [a for a in actions if a != "answer"]
        return actions

    @staticmethod
    def require_plan_after_rewind(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """After rewind, must plan again."""
        if state_sequence and state_sequence[-1].state == "rewind":
            filtered = [a for a in actions if a == "plan"]
            return filtered if filtered else actions
        return actions

    @staticmethod
    def break_on_consecutive_continue(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """Return empty list if two consecutive continue actions."""
        if state_sequence and len(state_sequence) >= 2:
            if state_sequence[-1].state == "continue" and state_sequence[-2].state == "continue":
                return []
        return actions

    @staticmethod
    def answer_after_continue(
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        actions: List[str]
    ) -> List[str]:
        """After continue, only keep the answer action."""
        if state_sequence and state_sequence[-1].state == "continue":
            filtered = [a for a in actions if a == "answer"]
            return filtered if filtered else actions
        return actions


class ActionRuleEnforcer:
    """Chains multiple rule functions to filter available actions.

    Example usage:
        enforcer = ActionRuleEnforcer(
            ActionRule.require_initial_plan_or_think,
            ActionRule.require_plan_before_execute,
        )
        available = enforcer.update_actions("initial", [], all_actions)
    """

    def __init__(self, *rules: Callable):
        """Initialize with rule functions.

        Args:
            *rules: Variable number of rule functions. Each function should
                    have signature (current_state, state_sequence, actions) -> actions.
        """
        self.rules = list(rules)

    def update_actions(
        self,
        current_state: str,
        state_sequence: Optional[List[StateResult]],
        full_actions: List[str]
    ) -> List[str]:
        """Apply all rules sequentially to filter available actions.

        Args:
            current_state: Current workflow state.
            state_sequence: Full sequence of prior StateResult objects.
            full_actions: Full list of action names to filter from.

        Returns:
            Filtered list of available action names.
        """
        actions = list(full_actions)
        for rule in self.rules:
            actions = rule(current_state, state_sequence, actions)
        return actions


# ============== Predefined Enforcers ==============

# Default enforcer for toolflow (function calling based)
DEFAULT_ENFORCER = ActionRuleEnforcer(
    ActionRule.require_initial_plan_or_think,
    ActionRule.require_continue_before_answer,
    ActionRule.break_on_consecutive_continue,
)

# Enforcer for treeflow (XML-based prompts)
TREEFLOW_ENFORCER = ActionRuleEnforcer(
    ActionRule.require_initial_plan_or_think,
    ActionRule.require_plan_before_execute,
    ActionRule.require_plan_after_rewind,
)

# Minimal enforcer - only requires initial plan or think
MINIMAL_ENFORCER = ActionRuleEnforcer(
    ActionRule.require_initial_plan_or_think,
)

# No rules enforcer - allows all actions
NO_RULES_ENFORCER = ActionRuleEnforcer()
