"""Tracking system for model interactions and content provenance."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import warnings

import numpy as np

if TYPE_CHECKING:
    from camel.memories import MemoryRecord


@dataclass
class ContentElement:
    """A single element in the content pool."""

    uid: int
    role: str
    content: str  # Content used for matching (may be str(tool_calls) if original was empty)
    original_content: str  # Original content for retrieval
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


class InteractionTracker:
    """Tracks content pool and model interactions across multiple LLMs.

    Content Pool:
        - Shared across all LLMs
        - Each (role, content) tuple is stored with a unique uid (1-indexed)

    Interaction Pool:
        - Each interaction records: llm_id, response_uid, context_uids
        - Export produces R×M matrix where R=num_interactions, M=pool_size
    """

    def __init__(self, tokenizer: Optional[AutoTokenizer] = None, concise_str: bool = False, sort_by_llm_id: bool = False):
        """Initialize tracker.

        Args:
            tokenizer: Optional HuggingFace tokenizer for apply_chat_template in get_message_text().
            concise_str: If True, __str__ groups elements with same content (different roles) into one column.
            sort_by_llm_id: If True, __str__ sorts interaction rows by LLM ID.
        """
        self._pool: list[ContentElement] = []
        self._uuid_to_uid: dict[str, int] = {}  # MemoryRecord.uuid -> uid for cross-LLM alignment
        self._content_to_uid: dict[tuple[str, str, Optional[str]], int] = {}
        self._interactions: list[tuple[int, int, list[int]]] = []  # (llm_id, response_uid, context_uids)
        self._edges: set[tuple[int, int]] = set()  # (from_uid, to_uid) for dependency graph
        self._tokenizer = tokenizer
        self._concise_str = concise_str
        self._sort_by_llm_id = sort_by_llm_id
        self._llm_tools: dict[int, list] = {}  # llm_id -> tools list
        self.state_sequence: list[str] = []
        self.state_results: list = []

    def record(self, messages: list[dict], llm_id: int = 0) -> int:
        """Record an interaction from a message list.

        Args:
            messages: List of message dicts with "role", "content", and optionally
                      "tool_calls", "tool_call_id" keys.
                      Last message must have role="assistant".
            llm_id: Identifier for the LLM that generated the response.

        Returns:
            interaction_id (0-indexed)

        Raises:
            ValueError: If messages is empty or last message is not from assistant.
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        if messages[-1].get("role") != "assistant":
            raise ValueError("Last message must be from assistant")

        uids = []
        uids_seen_in_this_call = set()  # Track UIDs used in this record() call
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]  # Content for matching (may be str(tool_calls))
            original_content = msg.get("original_content", content)  # Original content for retrieval
            tool_call_id = msg.get("tool_call_id", None)
            key = (role, content, tool_call_id)

            if key in self._content_to_uid:
                uid = self._content_to_uid[key]
                # If this UID was already used in this call, create a new one
                # This handles repeated messages (e.g., same prompt used multiple times)
                if uid in uids_seen_in_this_call:
                    uid = len(self._pool) + 1  # 1-indexed
                    self._pool.append(ContentElement(
                        uid=uid,
                        role=role,
                        content=content,
                        original_content=original_content,
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                    ))
                    # Don't update _content_to_uid - keep pointing to first occurrence
            else:
                uid = len(self._pool) + 1  # 1-indexed
                self._pool.append(ContentElement(
                    uid=uid,
                    role=role,
                    content=content,
                    original_content=original_content,
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                ))
                self._content_to_uid[key] = uid

            uids.append(uid)
            uids_seen_in_this_call.add(uid)

        response_uid = uids[-1]
        context_uids = uids[:-1]

        # Add edges for dependency graph (sequential order: A→B, B→C, C→D)
        for i in range(len(uids) - 1):
            self._edges.add((uids[i], uids[i + 1]))

        interaction_id = len(self._interactions)
        self._interactions.append((llm_id, response_uid, context_uids))

        return interaction_id

    def get_pool_size(self) -> int:
        """Return current pool size M."""
        return len(self._pool)

    def register_tools(self, llm_id: int, tools: list) -> None:
        """Register tools for a specific LLM.

        Converts FunctionTool objects to OpenAI tool schemas at registration time.

        Args:
            llm_id: The LLM identifier.
            tools: List of tools (FunctionTool objects or dicts) available to this LLM.
        """
        # Convert FunctionTool objects to schemas immediately
        tool_schemas = []
        for tool in tools:
            if hasattr(tool, 'get_openai_tool_schema'):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                tool_schemas.append(tool)
        self._llm_tools[llm_id] = tool_schemas

    def get_tools(self, llm_id: int) -> Optional[list]:
        """Get registered tool schemas for a specific LLM.

        Args:
            llm_id: The LLM identifier.

        Returns:
            List of tool schemas if registered, None otherwise.
        """
        return self._llm_tools.get(llm_id)

    def register_shared_records(self, records: list["MemoryRecord"]) -> None:
        """Register MemoryRecords being shared between agents for UUID-based alignment.

        When records are forwarded from one agent to another, call this method
        before writing them to the target agent's memory. This establishes
        UUID -> UID mapping so that when the target agent's interaction is recorded,
        shared messages are correctly aligned to existing UIDs (not treated as new).

        Args:
            records: List of MemoryRecords being shared. Each record's UUID will be
                     mapped to an existing UID (via content matching) or a new UID.
        """
        for record in records:
            uuid_str = str(record.uuid)
            if uuid_str in self._uuid_to_uid:
                continue  # Already registered

            role = record.role_at_backend.name.lower()
            content = record.message.content or ""
            key = (role, content)

            # Map UUID to existing UID if content matches, otherwise no mapping yet
            # (UID will be created when record() is called)
            if key in self._content_to_uid:
                self._uuid_to_uid[uuid_str] = self._content_to_uid[key]

    def get_uid_by_uuid(self, uuid_str: str) -> Optional[int]:
        """Get UID for a registered UUID.

        Args:
            uuid_str: The UUID string to look up.

        Returns:
            UID if found, None otherwise.
        """
        return self._uuid_to_uid.get(uuid_str)

    def _compute_tids(self) -> Optional[dict[int, int]]:
        """Compute topological order (uid -> tid) using Kahn's algorithm.

        Returns:
            dict mapping uid to tid, or None if cycle detected.
        """
        pool_size = len(self._pool)
        if pool_size == 0:
            return {}

        # Build in-degree and adjacency list
        in_degree = {uid: 0 for uid in range(1, pool_size + 1)}
        adj = {uid: [] for uid in range(1, pool_size + 1)}
        for from_uid, to_uid in self._edges:
            adj[from_uid].append(to_uid)
            in_degree[to_uid] += 1

        # Kahn's algorithm
        queue = [uid for uid, deg in in_degree.items() if deg == 0]
        uid_to_tid = {}
        tid = 1
        while queue:
            # Sort for deterministic order among nodes with same in-degree
            queue.sort()
            uid = queue.pop(0)
            uid_to_tid[uid] = tid
            tid += 1
            for neighbor in adj[uid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(uid_to_tid) != pool_size:
            return None  # Cycle detected
        return uid_to_tid

    # ANSI color codes for roles
    _ROLE_COLORS = {
        "system": "\033[95m",    # Magenta
        "user": "\033[92m",      # Green
        "assistant": "\033[94m", # Blue
        "tool": "\033[93m",      # Yellow
    }
    _RESET = "\033[0m"

    def __str__(self) -> str:
        """Return ASCII visualization of the interaction matrix with colored roles.
        
        Columns are ordered by topological order (tid). If a cycle is detected,
        warns and falls back to uid order.
        
        If concise_str=True, elements with same content (but different roles) are
        merged into one column with a shared tcid (topology content id).
        
        If sort_by_llm_id=True, interaction rows are sorted by LLM ID.
        """
        num_interactions = len(self._interactions)
        pool_size = len(self._pool)

        if num_interactions == 0:
            return "InteractionTracker: No interactions recorded."

        # Compute topological order
        uid_to_tid = self._compute_tids()
        if uid_to_tid is None:
            warnings.warn("Cycle detected in dependency graph, falling back to uid order")
            uid_to_tid = {uid: uid for uid in range(1, pool_size + 1)}  # fallback: tid = uid

        matrix = self.export_context_matrix()

        if self._concise_str:
            return self._str_concise(uid_to_tid, matrix)
        else:
            return self._str_full(uid_to_tid, matrix)

    def _get_interactions_for_display(self) -> list[tuple[int, tuple[int, int, list[int]]]]:
        """Get interactions with original indices, optionally sorted by LLM ID.
        
        Returns:
            List of (original_index, (llm_id, response_uid, context_uids)) tuples.
        """
        if self._sort_by_llm_id:
            return sorted(enumerate(self._interactions), key=lambda x: (x[1][0], x[0]))
        return list(enumerate(self._interactions))

    def _str_full(self, uid_to_tid: dict[int, int], matrix: np.ndarray) -> str:
        """Full visualization with one column per uid."""
        num_interactions = len(self._interactions)
        pool_size = len(self._pool)
        
        uid_order = sorted(range(1, pool_size + 1), key=lambda u: uid_to_tid[u])

        lines = ["Interaction Context Matrix (ordered by tid)", "=" * 40]

        # Legend
        legend_parts = [f"{self._ROLE_COLORS.get(r, '')}{r[0].upper()}{self._RESET}" 
                        for r in ["system", "user", "assistant", "tool"]]
        lines.append(f"Legend: {' '.join(legend_parts)} (S=system, U=user, A=assistant, T=tool)")
        lines.append("")

        # Header with tids in topological order
        tid_width = max(3, len(str(pool_size)))
        label_width = 20
        header = " " * label_width + "".join(f"{uid_to_tid[uid]:^{tid_width}}" for uid in uid_order)
        lines.append(header)
        lines.append(" " * label_width + "-" * (pool_size * tid_width))

        # Each interaction row
        for orig_i, (llm_id, response_uid, _) in self._get_interactions_for_display():
            response_tid = uid_to_tid[response_uid]
            label = f"I{orig_i} (LLM{llm_id})→T{response_tid}"
            row_parts = []
            for uid in uid_order:
                j = uid - 1  # 0-indexed for matrix/pool access
                if uid == response_uid:
                    row_parts.append(f"{self._ROLE_COLORS['assistant']}[R]{self._RESET}")
                elif matrix[orig_i, j]:
                    role = self._pool[j].role
                    color = self._ROLE_COLORS.get(role, "")
                    row_parts.append(f"[{color}■{self._RESET}]")
                else:
                    row_parts.append("[ ]")
            row = "".join(f"{part:^{tid_width}}" for part in row_parts)
            lines.append(f"{label:<{label_width}}{row}")

        lines.append("=" * 40)
        lines.append(f"Pool size: {pool_size}, Interactions: {num_interactions}")

        return "\n".join(lines)

    def _str_concise(self, uid_to_tid: dict[int, int], matrix: np.ndarray) -> str:
        """Concise visualization: group uids with same content into one column (tcid)."""
        num_interactions = len(self._interactions)
        pool_size = len(self._pool)

        # Group uids by content (ignoring role)
        content_to_uids: dict[str, list[int]] = {}
        for elem in self._pool:
            content_to_uids.setdefault(elem.content, []).append(elem.uid)

        # Assign tcid to each group (use min tid among group members for ordering)
        groups = list(content_to_uids.values())  # list of uid lists
        group_min_tid = [min(uid_to_tid[uid] for uid in g) for g in groups]
        sorted_groups = [g for _, g in sorted(zip(group_min_tid, groups))]
        
        # Build uid_to_tcid mapping
        uid_to_tcid: dict[int, int] = {}
        for tcid, group in enumerate(sorted_groups, start=1):
            for uid in group:
                uid_to_tcid[uid] = tcid

        num_tcids = len(sorted_groups)

        lines = ["Interaction Context Matrix (concise, ordered by tcid)", "=" * 50]

        # Legend
        legend_parts = [f"{self._ROLE_COLORS.get(r, '')}{r[0].upper()}{self._RESET}" 
                        for r in ["system", "user", "assistant", "tool"]]
        lines.append(f"Legend: {' '.join(legend_parts)} (S=system, U=user, A=assistant, T=tool)")
        lines.append("")

        # Header with tcids
        tcid_width = max(5, len(str(num_tcids)) + 2)
        label_width = 20
        header = " " * label_width + "".join(f"{tcid:^{tcid_width}}" for tcid in range(1, num_tcids + 1))
        lines.append(header)
        lines.append(" " * label_width + "-" * (num_tcids * tcid_width))

        # Each interaction row
        for orig_i, (llm_id, response_uid, context_uids) in self._get_interactions_for_display():
            response_tcid = uid_to_tcid[response_uid]
            label = f"I{orig_i} (LLM{llm_id})→C{response_tcid}"
            row_parts = []

            for tcid, group in enumerate(sorted_groups, start=1):
                # Collect roles present in this tcid for this interaction
                roles_present = []
                is_response = False
                for uid in group:
                    if uid == response_uid:
                        is_response = True
                    elif matrix[orig_i, uid - 1]:  # uid is in context
                        role = self._pool[uid - 1].role
                        if role not in roles_present:
                            roles_present.append(role)

                if is_response:
                    # Show response indicator with any context roles from same content
                    if roles_present:
                        role_chars = "".join(
                            f"{self._ROLE_COLORS.get(r, '')}{r[0].upper()}{self._RESET}"
                            for r in roles_present
                        )
                        row_parts.append(f"{role_chars}{self._ROLE_COLORS['assistant']}R{self._RESET}")
                    else:
                        row_parts.append(f"{self._ROLE_COLORS['assistant']}[R]{self._RESET}")
                elif roles_present:
                    # Show all roles present with their colors
                    role_chars = "".join(
                        f"{self._ROLE_COLORS.get(r, '')}{r[0].upper()}{self._RESET}"
                        for r in roles_present
                    )
                    row_parts.append(f"[{role_chars}]")
                else:
                    row_parts.append("[ ]")

            row = "".join(f"{part:^{tcid_width}}" for part in row_parts)
            lines.append(f"{label:<{label_width}}{row}")

        lines.append("=" * 50)
        lines.append(f"Pool size: {pool_size}, Content groups: {num_tcids}, Interactions: {num_interactions}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"InteractionTracker(pool_size={len(self._pool)}, interactions={len(self._interactions)})"

    def get_messages(self, llm_id: int = 0, response_uid: Optional[int] = None) -> list[dict]:
        """Get messages for an interaction by LLM ID and response UID.

        Returns messages in HF tokenizer compatible format, including tool_calls
        and tool_call_id when present.

        Args:
            llm_id: The LLM identifier.
            response_uid: The response UID. If None, returns messages for the last response of this LLM.

        Returns:
            List of message dicts with 'role', 'content', and optionally 'tool_calls', 'tool_call_id'.

        Raises:
            ValueError: If no matching interaction is found.
        """
        context_uids = None
        target_response_uid = response_uid

        if response_uid is None:
            # Find last interaction for this LLM
            for lid, rid, ctx in reversed(self._interactions):
                if lid == llm_id:
                    target_response_uid = rid
                    context_uids = ctx
                    break
        else:
            # Find interaction by response_uid and llm_id
            for lid, rid, ctx in self._interactions:
                if lid == llm_id and rid == response_uid:
                    context_uids = ctx
                    break

        if context_uids is None:
            if response_uid is None:
                raise ValueError(f"No interactions found for llm_id={llm_id}")
            raise ValueError(f"No interaction found for llm_id={llm_id}, response_uid={response_uid}")

        # Build messages list: context + response
        all_uids = context_uids + [target_response_uid]
        messages = []
        for uid in all_uids:
            elem = self._pool[uid - 1]
            msg = {"role": elem.role, "content": elem.original_content}
            if elem.tool_calls is not None:
                msg["tool_calls"] = elem.tool_calls
            if elem.tool_call_id is not None:
                msg["tool_call_id"] = elem.tool_call_id
            messages.append(msg)

        return messages

    def get_message_text(self, llm_id: int = 0, response_uid: Optional[int] = None) -> str:
        """Get messages as formatted text for an interaction.

        If tokenizer was provided at init, uses apply_chat_template.
        Otherwise, converts messages to a simple text format.
        If tools were registered for this llm_id, they are included in the template.

        Args:
            llm_id: The LLM identifier.
            response_uid: The response UID. If None, returns text for the last response of this LLM.

        Returns:
            Formatted text string of the conversation.
        """
        messages = self.get_messages(llm_id=llm_id, response_uid=response_uid)

        if self._tokenizer is not None:
            template_kwargs = {}
            tool_schemas = self._llm_tools.get(llm_id)
            if tool_schemas:
                template_kwargs["tools"] = tool_schemas
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )
        else:
            # Simple text format without tokenizer
            lines = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                lines.append(f"[{role}]: {content}")
            return "\n".join(lines)

    def export_context_matrix(self, llm_id: Optional[int] = None) -> np.ndarray:
        """Export context dependency matrix.

        Args:
            llm_id: If provided, only export interactions from this LLM.
                    If None, export all interactions.

        Returns:
            bool array of shape (num_interactions, pool_size)
            matrix[i, j] = True if interaction i used element (j+1) as context
        """
        pool_size = len(self._pool)

        if llm_id is None:
            interactions = self._interactions
        else:
            interactions = [(lid, rid, ctx) for lid, rid, ctx in self._interactions if lid == llm_id]

        num_interactions = len(interactions)
        matrix = np.zeros((num_interactions, pool_size), dtype=bool)

        for i, (_, _, context_uids) in enumerate(interactions):
            for uid in context_uids:
                matrix[i, uid - 1] = True

        return matrix

    def export_response_uids(self, llm_id: Optional[int] = None) -> list[int]:
        """Return list of response uids, one per interaction.

        Args:
            llm_id: If provided, only export from this LLM. If None, export all.
        """
        if llm_id is None:
            return [response_uid for _, response_uid, _ in self._interactions]
        return [response_uid for lid, response_uid, _ in self._interactions if lid == llm_id]

    def export_llm_ids(self) -> list[int]:
        """Return list of llm_ids, one per interaction."""
        return [llm_id for llm_id, _, _ in self._interactions]

    def get_unique_llm_ids(self) -> list[int]:
        """Return list of unique LLM IDs that have recorded interactions."""
        return list({llm_id for llm_id, _, _ in self._interactions})

    def get_uids(self, llm_id: int) -> list[int]:
        """Return all unique UIDs (context + response) for a given LLM.

        Args:
            llm_id: The LLM identifier.

        Returns:
            Sorted list of unique UIDs associated with this LLM.
        """
        uids = set()
        for lid, response_uid, context_uids in self._interactions:
            if lid == llm_id:
                uids.add(response_uid)
                uids.update(context_uids)
        return sorted(uids)

    def message_to_uid(self, message: dict) -> int:
        """Get the UID for a single message dict.

        Args:
            message: Message dict with 'role' and 'content' keys.

        Returns:
            UID if found, -1 otherwise.
        """
        role = message.get("role", "")
        content = message.get("content", "")
        tool_call_id = message.get("tool_call_id", None)
        if not content and "tool_calls" in message:
            content = str(message["tool_calls"])
        return self._content_to_uid.get((role, content, tool_call_id), -1)

    def messages_to_uids(self, messages: list[dict]) -> list[int]:
        """Get UIDs for a list of message dicts.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            List of UIDs. -1 for messages not found in the pool.
        """
        return [self.message_to_uid(msg) for msg in messages]

    def plot(self, path: str = "interaction.jpg") -> None:
        """Plot and save the interaction matrix with each interaction as a subplot.

        Args:
            path: Output file path. Default: "interaction.jpg"
        """
        num_interactions = len(self._interactions)
        pool_size = len(self._pool)

        if num_interactions == 0:
            print("No interactions to plot.")
            return

        matrix = self.export_context_matrix()

        # Create subplots: one row per interaction
        fig, axes = plt.subplots(
            num_interactions, 1,
            figsize=(max(8, pool_size * 0.3), num_interactions * 0.8 + 1),
            squeeze=False,
        )

        for i, (llm_id, response_uid, _) in enumerate(self._interactions):
            ax = axes[i, 0]
            ax.imshow(matrix[i:i+1, :], cmap="Blues", aspect="auto", vmin=0, vmax=1)

            ax.set_yticks([0])
            ax.set_yticklabels([f"I{i} (LLM{llm_id})→R{response_uid}"])
            ax.set_xticks(range(pool_size))
            ax.set_xticklabels(range(1, pool_size + 1), fontsize=8)

            # Add grid
            ax.set_xticks(np.arange(-0.5, pool_size, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)

        axes[-1, 0].set_xlabel("Content Pool UID")
        fig.suptitle("Interaction Context Matrix", fontsize=12)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved interaction plot to {path}")


def record_interaction(
    tracker: Optional[InteractionTracker],
    messages: list[dict],
    llm_id: int = 0,
) -> Optional[int]:
    """Record an interaction if tracker is provided.

    Wrapper function that handles None tracker and normalizes messages.
    For messages with empty content but 'tool_calls', uses str(tool_calls) as content
    for deduplication, while preserving original tool_calls for retrieval.

    Args:
        tracker: InteractionTracker instance or None (no-op if None)
        messages: List of message dicts with 'role' and 'content' keys
        llm_id: Identifier for the LLM

    Returns:
        interaction_id if tracked, None if tracker is None
    """
    if tracker is None:
        return None

    clean_messages = []
    for msg in messages:
        role = msg["role"]
        original_content = msg.get("content", "")
        content = original_content  # Content for matching

        # If content is empty but tool_calls exists, use tool_calls as content for matching
        if not content and "tool_calls" in msg:
            content = str(msg["tool_calls"])

        # Build clean message
        clean_msg = {
            "role": role,
            "content": content,
            "original_content": original_content,
        }

        # Preserve original tool fields
        if "tool_calls" in msg:
            clean_msg["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            clean_msg["tool_call_id"] = msg["tool_call_id"]

        clean_messages.append(clean_msg)

    return tracker.record(clean_messages, llm_id=llm_id)


@dataclass
class TreeNode:
    """A node in the tree structure."""
    step_idx: int
    parent_idx: Optional[int]
    action: str  # execute, plan, think, rewind, exam, answer
    data: dict = field(default_factory=dict)
    children: list[int] = field(default_factory=list)


class TreeTracker:
    """Tracks tree structure for tree-based research workflow."""

    def __init__(self):
        self._nodes: dict[int, TreeNode] = {}  # node_id -> TreeNode
        self._rewinds: list[tuple[int, int, str]] = []  # (from_node_id, to_step_idx, summary)
        self._current_path: list[int] = []  # list of node_ids in current execution path
        self._step_to_node: dict[int, int] = {}  # step_idx -> node_id for execute nodes
        self._tools_per_round: list[list[str]] = []  # tools used per round

    def get_current_parent(self) -> Optional[int]:
        """Return the node_id of the current parent (last node in path), or None if empty."""
        return self._current_path[-1] if self._current_path else None

    def add_node(self, node_id: int, parent_id: Optional[int], action: str, data: dict = None):
        """Add a node to the tree.

        Args:
            node_id: Unique identifier for this node (monotonically increasing).
            parent_id: Node ID of parent (None for root).
            action: Action type (execute, plan, think, rewind, exam, answer).
            data: Associated data (task, tasks, answer, etc.).
        """
        node = TreeNode(
            step_idx=node_id,  # Using node_id as step_idx for compatibility
            parent_idx=parent_id,
            action=action,
            data=data or {}
        )
        self._nodes[node_id] = node

        # Update parent's children
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children.append(node_id)

        # Update current path
        self._current_path.append(node_id)

    def mark_rewind(self, to_step_idx: int, summary: str):
        """Record a rewind event and update current path.

        Args:
            to_step_idx: Step index to rewind to (execution position, not node_id).
            summary: Summary of the failed path.
        """
        from_node_id = self._current_path[-1] if self._current_path else 0
        self._rewinds.append((from_node_id, to_step_idx, summary))
        # Find the node at to_step_idx and trim path to that point
        if to_step_idx in self._step_to_node:
            target_node_id = self._step_to_node[to_step_idx]
            if target_node_id in self._current_path:
                idx = self._current_path.index(target_node_id)
                self._current_path = self._current_path[:idx + 1]

    def register_step(self, step_idx: int, node_id: int):
        """Register mapping from step_idx to node_id for execute nodes."""
        self._step_to_node[step_idx] = node_id

    def record_tools_used(self, tools: list[str]):
        """Record tools used in the current round."""
        self._tools_per_round.append(tools if tools else [])

    def get_tools_per_round(self) -> list[list[str]]:
        """Return list of tools used per round."""
        return list(self._tools_per_round)

    def get_all_tools_used(self) -> list[str]:
        """Return flat list of all unique tools used across all rounds."""
        all_tools = []
        for tools in self._tools_per_round:
            for tool in tools:
                if tool not in all_tools:
                    all_tools.append(tool)
        return all_tools

    def get_current_path(self) -> list[int]:
        """Return the current path from root to current node."""
        return list(self._current_path)

    def get_node(self, step_idx: int) -> Optional[TreeNode]:
        """Get node by step index."""
        return self._nodes.get(step_idx)

    def __str__(self) -> str:
        """Return ASCII visualization of the tree."""
        if not self._nodes:
            return "TreeTracker: Empty"

        lines = ["Tree Structure", "=" * 40]

        def format_node(idx: int, prefix: str = "", is_last: bool = True) -> list[str]:
            node = self._nodes.get(idx)
            if not node:
                return []

            connector = "└── " if is_last else "├── "
            data_str = ""
            if node.action == "execute" and "task" in node.data:
                data_str = f": {node.data['task'][:50]}..."
            elif node.action == "answer" and "answer" in node.data:
                data_str = f": {node.data['answer'][:50]}..."

            result = [f"{prefix}{connector}[{idx}] {node.action}{data_str}"]

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_idx in enumerate(node.children):
                is_child_last = (i == len(node.children) - 1)
                result.extend(format_node(child_idx, child_prefix, is_child_last))

            return result

        # Find root nodes (nodes without parent or parent not in tree)
        roots = [idx for idx, node in self._nodes.items()
                 if node.parent_idx is None or node.parent_idx not in self._nodes]

        for root_idx in sorted(roots):
            lines.extend(format_node(root_idx, "", True))

        if self._rewinds:
            lines.append("")
            lines.append("Rewinds:")
            for from_node_id, to_step_idx, summary in self._rewinds:
                lines.append(f"  node {from_node_id} → step {to_step_idx}: {summary[:60]}...")

        lines.append("=" * 40)
        lines.append(f"Nodes: {len(self._nodes)}, Rewinds: {len(self._rewinds)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TreeTracker(nodes={len(self._nodes)}, rewinds={len(self._rewinds)})"
