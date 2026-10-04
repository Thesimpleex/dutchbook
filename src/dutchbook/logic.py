"""Propositional formulas: parsing, printing, evaluation and world enumeration.

Grammar (binding strength from tight to loose)::

    not    ~  !  ¬  not
    and    &  &&  ∧  and
    or     |  ||  ∨  or
    implies  ->  =>  →  implies      (right associative)
    iff    <->  <=>  ↔  iff          (associative)

Atoms are identifiers; ``true`` / ``false`` (or ``⊤`` / ``⊥``) are constants.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_ATOMS = 22
"""Largest number of atoms for which worlds are enumerated (2**22 worlds)."""

_UNICODE = {"not": "¬", "and": "∧", "or": "∨", "implies": "→", "iff": "↔"}
_ASCII = {"not": "~", "and": "&", "or": "|", "implies": "->", "iff": "<->"}
_PRECEDENCE = {"iff": 0, "implies": 1, "or": 2, "and": 3, "not": 4}


class FormulaSyntaxError(ValueError):
    """Raised when a formula string cannot be parsed."""


class Formula:
    """Base class of all propositional formulas (immutable, hashable)."""

    __slots__ = ()

    def atoms(self) -> tuple[str, ...]:
        """Atom names in order of first appearance."""
        seen: dict[str, None] = {}
        self._collect(seen)
        return tuple(seen)

    def evaluate(self, values: Mapping[str, Any]) -> np.ndarray:
        """Truth value under an assignment; entries of ``values`` may be boolean arrays.

        Evaluation is vectorised: with arrays of length ``n`` the result is a boolean
        array of length ``n`` (one entry per world).
        """
        return np.asarray(self._eval(values), dtype=bool)

    def ascii(self) -> str:
        """Round-trippable ASCII rendering."""
        return _render(self, _ASCII, 0, "")

    def iff(self, other: Formula) -> Formula:
        return Iff(self, other)

    def __str__(self) -> str:
        return _render(self, _UNICODE, 0, "")

    def __invert__(self) -> Formula:
        return Not(self)

    def __and__(self, other: Formula) -> Formula:
        return And(self, other)

    def __or__(self, other: Formula) -> Formula:
        return Or(self, other)

    def __rshift__(self, other: Formula) -> Formula:
        return Implies(self, other)

    def _collect(self, seen: dict[str, None]) -> None:
        raise NotImplementedError

    def _eval(self, values: Mapping[str, Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Atom(Formula):
    name: str

    def _collect(self, seen: dict[str, None]) -> None:
        seen.setdefault(self.name)

    def _eval(self, values: Mapping[str, Any]) -> Any:
        try:
            return values[self.name]
        except KeyError:
            raise KeyError(f"no truth value supplied for atom {self.name!r}") from None


@dataclass(frozen=True, slots=True)
class Const(Formula):
    value: bool

    def _collect(self, seen: dict[str, None]) -> None:
        return None

    def _eval(self, values: Mapping[str, Any]) -> Any:
        return np.bool_(self.value)


@dataclass(frozen=True, slots=True)
class Not(Formula):
    arg: Formula

    def _collect(self, seen: dict[str, None]) -> None:
        self.arg._collect(seen)

    def _eval(self, values: Mapping[str, Any]) -> Any:
        return ~np.asarray(self.arg._eval(values), dtype=bool)


@dataclass(frozen=True, slots=True)
class _Binary(Formula):
    left: Formula
    right: Formula

    def _collect(self, seen: dict[str, None]) -> None:
        self.left._collect(seen)
        self.right._collect(seen)


@dataclass(frozen=True, slots=True)
class And(_Binary):
    def _eval(self, values: Mapping[str, Any]) -> Any:
        return np.asarray(self.left._eval(values), dtype=bool) & np.asarray(
            self.right._eval(values), dtype=bool
        )


@dataclass(frozen=True, slots=True)
class Or(_Binary):
    def _eval(self, values: Mapping[str, Any]) -> Any:
        return np.asarray(self.left._eval(values), dtype=bool) | np.asarray(
            self.right._eval(values), dtype=bool
        )


@dataclass(frozen=True, slots=True)
class Implies(_Binary):
    def _eval(self, values: Mapping[str, Any]) -> Any:
        return ~np.asarray(self.left._eval(values), dtype=bool) | np.asarray(
            self.right._eval(values), dtype=bool
        )


@dataclass(frozen=True, slots=True)
class Iff(_Binary):
    def _eval(self, values: Mapping[str, Any]) -> Any:
        return np.asarray(self.left._eval(values), dtype=bool) == np.asarray(
            self.right._eval(values), dtype=bool
        )


_BINARY_KIND: dict[type, str] = {And: "and", Or: "or", Implies: "implies", Iff: "iff"}


def _render(f: Formula, symbols: Mapping[str, str], parent: int, side: str) -> str:
    """Render with the fewest parentheses that preserve the parse tree.

    ``side`` is ``"l"`` or ``"r"`` for an operand of a binary operator, so that equal
    precedence is parenthesised on the side opposite to the operator's associativity.
    """
    if isinstance(f, Atom):
        return f.name
    if isinstance(f, Const):
        return (
            ("⊤" if f.value else "⊥") if symbols is _UNICODE else ("true" if f.value else "false")
        )
    if isinstance(f, Not):
        return symbols["not"] + _render(f.arg, symbols, _PRECEDENCE["not"], "")
    kind = _BINARY_KIND[type(f)]
    prec = _PRECEDENCE[kind]
    assert isinstance(f, _Binary)
    left = _render(f.left, symbols, prec, "l")
    right = _render(f.right, symbols, prec, "r")
    text = f"{left} {symbols[kind]} {right}"
    right_assoc = kind == "implies"
    needs = prec < parent or (prec == parent and (side == "l" if right_assoc else side == "r"))
    return f"({text})" if needs else text


_TOKEN = re.compile(
    r"""\s*(?:
        (?P<iff><->|<=>|↔|≡)
      | (?P<implies>->|=>|→|⊃)
      | (?P<and>&&|&|∧)
      | (?P<or>\|\||\||∨)
      | (?P<not>~|!|¬)
      | (?P<lpar>\()
      | (?P<rpar>\))
      | (?P<top>⊤)
      | (?P<bot>⊥)
      | (?P<word>[^\W\d]\w*)
    )""",
    re.VERBOSE,
)
_WORDS = {
    "and": "and",
    "or": "or",
    "not": "not",
    "implies": "implies",
    "iff": "iff",
    "true": "top",
    "false": "bot",
}


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            bad = pos + len(text[pos:]) - len(text[pos:].lstrip())
            raise FormulaSyntaxError(_message(text, bad, f"unexpected character {text[bad]!r}"))
        kind = m.lastgroup or ""
        value = m.group(kind)
        start = m.start(kind)
        if kind == "word":
            kind = _WORDS.get(value.lower(), "atom")
        tokens.append((kind, value, start))
        pos = m.end()
    return tokens


def _message(text: str, pos: int, problem: str) -> str:
    return f"{problem} at position {pos}\n  {text}\n  {' ' * pos}^"


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i][0] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str, int]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def fail(self, problem: str) -> FormulaSyntaxError:
        pos = self.tokens[self.i][2] if self.i < len(self.tokens) else len(self.text)
        return FormulaSyntaxError(_message(self.text, pos, problem))

    def parse(self) -> Formula:
        if not self.tokens:
            raise FormulaSyntaxError("empty formula")
        f = self.iff()
        if self.i != len(self.tokens):
            raise self.fail(f"unexpected {self.tokens[self.i][1]!r}")
        return f

    def iff(self) -> Formula:
        f = self.implies()
        while self.peek() == "iff":
            self.take()
            f = Iff(f, self.implies())
        return f

    def implies(self) -> Formula:
        f = self.disjunction()
        if self.peek() == "implies":
            self.take()
            return Implies(f, self.implies())
        return f

    def disjunction(self) -> Formula:
        f = self.conjunction()
        while self.peek() == "or":
            self.take()
            f = Or(f, self.conjunction())
        return f

    def conjunction(self) -> Formula:
        f = self.negation()
        while self.peek() == "and":
            self.take()
            f = And(f, self.negation())
        return f

    def negation(self) -> Formula:
        if self.peek() == "not":
            self.take()
            return Not(self.negation())
        return self.primary()

    def primary(self) -> Formula:
        kind = self.peek()
        if kind == "atom":
            return Atom(self.take()[1])
        if kind == "top":
            self.take()
            return Const(True)
        if kind == "bot":
            self.take()
            return Const(False)
        if kind == "lpar":
            self.take()
            f = self.iff()
            if self.peek() != "rpar":
                raise self.fail("expected ')'")
            self.take()
            return f
        if kind is None:
            raise self.fail("unexpected end of formula")
        raise self.fail(f"unexpected {self.tokens[self.i][1]!r}")


def parse(text: str) -> Formula:
    """Parse a propositional formula.

    Raises
    ------
    FormulaSyntaxError
        With the offending position marked.
    """
    return _Parser(text).parse()


def as_formula(f: Formula | str) -> Formula:
    return parse(f) if isinstance(f, str) else f


def collect_atoms(formulas: Iterable[Formula]) -> tuple[str, ...]:
    """Union of atoms in order of first appearance."""
    seen: dict[str, None] = {}
    for f in formulas:
        f._collect(seen)
    return tuple(seen)


def enumerate_worlds(n_atoms: int) -> np.ndarray:
    """All ``2**n_atoms`` truth assignments as a boolean array ``(2**n, n)``.

    Rows are ordered like a truth table with the first atom varying slowest and
    ``True`` before ``False`` (so the first row makes every atom true).
    """
    if n_atoms < 0:
        raise ValueError("n_atoms must be non-negative")
    if n_atoms > MAX_ATOMS:
        raise ValueError(
            f"{n_atoms} atoms exceed the enumeration limit of {MAX_ATOMS}; "
            "use column generation (dutchbook.oracle) for larger problems"
        )
    idx = np.arange(1 << n_atoms, dtype=np.uint32)
    cols = [((idx >> np.uint32(n_atoms - 1 - i)) & np.uint32(1)) == 0 for i in range(n_atoms)]
    if not cols:
        return np.zeros((1, 0), dtype=bool)
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class WorldSpace:
    """Distinct truth patterns of a list of formulas.

    Attributes
    ----------
    formulas, atoms
        The formulas and the atoms they mention.
    W : ndarray, shape (k, m)
        One row per *distinct* truth pattern over the ``m`` formulas (0/1 floats).
        Worlds that agree on every formula are equivalent for coherence and are merged.
    assignments : ndarray of bool, shape (k, n_atoms)
        A representative truth assignment for every row of ``W``.
    """

    formulas: tuple[Formula, ...]
    atoms: tuple[str, ...]
    W: np.ndarray
    assignments: np.ndarray

    def describe_world(self, row: int) -> str:
        """Representative assignment as a conjunction of literals, e.g. ``A ∧ ¬B``."""
        if not self.atoms:
            return "⊤"
        lits = [a if v else f"¬{a}" for a, v in zip(self.atoms, self.assignments[row], strict=True)]
        return " ∧ ".join(lits)


def pattern_keys(W: np.ndarray) -> np.ndarray:
    """One hashable key per row: a packed integer for up to 63 columns, else raw bytes."""
    m = W.shape[1]
    if m <= 63:
        weights = np.left_shift(np.uint64(1), np.arange(m, dtype=np.uint64))
        return (W.astype(np.uint64) @ weights).astype(np.uint64)
    packed = np.packbits(W.astype(np.uint8), axis=1)
    return np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()


def world_space(
    formulas: Sequence[Formula | str],
    atoms: Sequence[str] | None = None,
) -> WorldSpace:
    """Enumerate worlds and the truth-value matrix of ``formulas``.

    Parameters
    ----------
    formulas
        Formulas or strings in the syntax of :func:`parse`.
    atoms
        Atom order. Defaults to order of first appearance; must cover every atom used.
    """
    fs = tuple(as_formula(f) for f in formulas)
    found = collect_atoms(fs)
    if atoms is None:
        atom_names = found
    else:
        atom_names = tuple(atoms)
        if len(set(atom_names)) != len(atom_names):
            raise ValueError("duplicate atom names")
        missing = [a for a in found if a not in atom_names]
        if missing:
            raise ValueError(f"formulas use atoms missing from `atoms`: {missing}")
    worlds = enumerate_worlds(len(atom_names))
    values = {a: worlds[:, i] for i, a in enumerate(atom_names)}
    full = np.empty((worlds.shape[0], len(fs)), dtype=bool)
    for j, f in enumerate(fs):
        full[:, j] = np.broadcast_to(f.evaluate(values), (worlds.shape[0],))
    keys = pattern_keys(full)
    _, first = np.unique(keys, return_index=True)
    first.sort()
    return WorldSpace(
        formulas=fs,
        atoms=atom_names,
        W=full[first].astype(float),
        assignments=worlds[first],
    )
