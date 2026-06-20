import re
from pathlib import Path
from typing import Dict, Set, Tuple


class DFA:
    """
    Minimal DFA runner that constructs DFA from Graphviz .dot files of the form:

        s0 [label="s0", shape=doublecircle];
        s1 [label="s1"];
        s0 -> s1 [label="("];
        __start0 -> s0 [label=""];

    Assumptions:
    - Accepting states are marked by shape=doublecircle.
    - The initial state is the target of the __start0 edge.
    - Edge labels are single-symbol alphabet entries, e.g. "(" and ")".
    """

    def __init__(self, dot_path: str):
        self.dot_path = Path(dot_path)

        self.states: Set[str] = set()
        self.accept_states: Set[str] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str], str] = {}
        self.start_state: str | None = None

        self._read_dot_file()
        self._sanity_check()

        if self.start_state is None:
            raise ValueError("No start state found. Expected an edge like: __start0 -> s0 [label=\"\"];")

    def _read_dot_file(self) -> None:
        text = self.dot_path.read_text()

        # Match node declarations like:
        # s0 [label="s0", shape=doublecircle];
        # s1 [label="s1"];
        node_pattern = re.compile(
            r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*?)\]\s*;',
            re.MULTILINE,
        )

        # Match edges like:
        # s0 -> s1 [label="("];
        # __start0 -> s0 [label=""];
        edge_pattern = re.compile(
            r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*'
            r'\[(.*?)\]\s*;',
            re.MULTILINE,
        )

        for state, attrs in node_pattern.findall(text):
            # Skip Graphviz helper start node, if present
            if state.startswith("__start"):
                continue

            self.states.add(state)

            if "shape=doublecircle" in attrs.replace(" ", ""):
                self.accept_states.add(state)

        for src, dst, attrs in edge_pattern.findall(text):
            label = self._extract_label(attrs)

            # Initial-state pseudo-edge
            if src.startswith("__start"):
                self.start_state = dst
                continue
            
            if src not in self.states or dst not in self.states:
                raise ValueError(f"Edge references undefined state: {src} -> {dst}")

            if label == "":
                raise ValueError(f"Unexpected empty transition label on edge {src} -> {dst}")

            key = (src, label)

            if key in self.transitions:
                old_dst = self.transitions[key]
                raise ValueError(
                    f"Nondeterministic transition found: "
                    f"{src} --{label}--> {old_dst} and {src} --{label}--> {dst}"
                )

            self.transitions[key] = dst
            self.alphabet.add(label)

    def _sanity_check(self) -> None:
        # sanity check: every src state has len(self.alphabet) outgoing edges
        for state in self.states:
            outgoing_labels = {label for (src, label), dst in self.transitions.items() if src == state}
            if outgoing_labels != self.alphabet:
                missing = self.alphabet - outgoing_labels
                raise ValueError(
                    f"State {state} is missing transitions for symbols: {sorted(missing)!r}"
                )
        # sanity check: start state is not None
        assert self.start_state is not None, "Start state should not be None after initialization."

    @staticmethod
    def _extract_label(attrs: str) -> str:
        """
        Extract label="..." from a DOT attribute string.

        Handles simple escaped characters inside labels.
        """
        match = re.search(r'label\s*=\s*"((?:\\.|[^"\\])*)"', attrs)
        if match is None:
            raise ValueError(f"Missing label in attrs: {attrs}")

        raw_label = match.group(1)

        # Decode common DOT-style escapes.
        return bytes(raw_label, "utf-8").decode("unicode_escape")

    def predict(self, x: str) -> bool:
        """
        Run the DFA on input string x.

        Returns:
            True  if accepted
            False if rejected

        """
        state = self.start_state

        for symbol in x:
            if symbol not in self.alphabet:
                raise ValueError(
                    f"Symbol {symbol!r} is not in DFA alphabet {sorted(self.alphabet)!r}"
                )

            key = (state, symbol)
            state = self.transitions[key]

        return state in self.accept_states

    def __repr__(self) -> str:
        return (
            f"DFA("
            f"num_states={len(self.states)}, "
            f"alphabet={sorted(self.alphabet)!r}, "
            f"start_state={self.start_state!r}, "
            f"accept_states={sorted(self.accept_states)!r}"
            f")"
        )