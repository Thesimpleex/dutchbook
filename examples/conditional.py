"""Conditional forecasts and called-off bets: the chain rule as a Dutch book."""

from dutchbook import check_conditional_coherence, find_conditional_dutch_book

events = ["A", "B given A", "A & B"]
for z in (0.20, 0.35):  # P(A) = 0.5 and P(B | A) = 0.4 force P(A and B) = 0.20
    q = [0.5, 0.4, z]
    verdict = check_conditional_coherence(events, q)
    print(f"P(A & B) = {z:.2f}: coherent = {verdict.coherent}")
    book = find_conditional_dutch_book(events, q)
    if book is not None:
        stakes = {str(e): round(float(s), 3) for e, s in zip(events, book.stakes, strict=True)}
        print(f"  stakes {stakes}, gain in every world where a bet stays on: {book.profit:.3f}")
