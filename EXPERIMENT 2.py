
import pandas as pd
from itertools import product

# Load dataset
data = pd.read_csv("EnjoySport.csv")
data.columns = data.columns.str.strip()

target_col = "EnjoySport"
attributes = [c for c in data.columns if c != target_col]

X = data[attributes].astype(str).values.tolist()
y = data[target_col].astype(str).str.strip().str.lower().tolist()

# Attribute domains
domains = [
    sorted(data[col].astype(str).unique().tolist())
    for col in attributes
]

# 'Ø' represents the most specific / empty constraint
BOTTOM = "Ø"
ANY = "?"

def covers(h, x):
    return all(a == ANY or a == b for a, b in zip(h, x))

def more_general(h1, h2):
    """Is h1 at least as general as h2?"""
    return all(
        a == ANY or a == b or b == BOTTOM
        for a, b in zip(h1, h2)
    )

def minimal_generalizations(h, x):
    result = list(h)
    for i in range(len(h)):
        if result[i] == BOTTOM:
            result[i] = x[i]
        elif result[i] != x[i] and result[i] != ANY:
            result[i] = ANY
    return [tuple(result)]

def minimal_specializations(g, x, S):
    results = []
    for i in range(len(g)):
        if g[i] == ANY:
            for value in domains[i]:
                if value != x[i]:
                    h = list(g)
                    h[i] = value
                    h = tuple(h)
                    if any(more_general(h, s) for s in S):
                        results.append(h)
    return results

def remove_more_general(hypotheses):
    # Keep only maximally specific boundary members
    return [
        h for h in hypotheses
        if not any(
            h != other and more_general(h, other)
            for other in hypotheses
        )
    ]

def remove_more_specific(hypotheses):
    # Keep only maximally general boundary members
    return [
        h for h in hypotheses
        if not any(
            h != other and more_general(other, h)
            for other in hypotheses
        )
    ]

def candidate_elimination(X, y):
    n = len(X[0])
    S = {tuple([BOTTOM] * n)}
    G = {tuple([ANY] * n)}

    for i, (x, label) in enumerate(zip(X, y), start=1):
        if label in ("yes", "true", "1"):
            # Remove G hypotheses that reject this positive example
            G = {g for g in G if covers(g, x)}

            new_S = set()
            for s in S:
                if covers(s, x):
                    new_S.add(s)
                else:
                    for h in minimal_generalizations(s, x):
                        if any(more_general(g, h) for g in G):
                            new_S.add(h)

            S = set(remove_more_general(list(new_S)))

        elif label in ("no", "false", "0"):
            # Remove S hypotheses that cover this negative example
            S = {s for s in S if not covers(s, x)}

            new_G = set()
            for g in G:
                if not covers(g, x):
                    new_G.add(g)
                else:
                    new_G.update(minimal_specializations(g, x, S))

            G = set(remove_more_specific(list(new_G)))

        else:
            raise ValueError(f"Unknown class label: {label}")

        print(f"\nExample {i}: {x} -> {label}")
        print("S =", sorted(map(list, S)))
        print("G =", sorted(map(list, G)))

    print("\nFinal Specific Boundary (S):")
    for h in sorted(S):
        print(list(h))

    print("\nFinal General Boundary (G):")
    for h in sorted(G):
        print(list(h))

candidate_elimination(X, y)
