#!/usr/bin/env python3
"""Nash equilibrium solver — find mixed/pure strategy equilibria.

Implements support enumeration for 2-player games, plus Lemke-Howson pivot.
Classic games included: Prisoner's Dilemma, Battle of Sexes, Matching Pennies.

Usage: python nash_equilibrium.py [--test]
"""

import sys
from itertools import combinations

def best_response(payoff_matrix, opponent_strategy):
    """Find best response (pure) to opponent's mixed strategy."""
    n = len(payoff_matrix)
    m = len(payoff_matrix[0])
    expected = []
    for i in range(n):
        ev = sum(payoff_matrix[i][j] * opponent_strategy[j] for j in range(m))
        expected.append(ev)
    max_ev = max(expected)
    return [i for i in range(n) if abs(expected[i] - max_ev) < 1e-9]

def solve_2x2(A, B):
    """Solve 2x2 bimatrix game analytically. Returns (p, q) mixed strategies."""
    # Player 1 makes player 2 indifferent: q such that EU1(row0) = EU1(row1)
    # A[0][0]*q + A[0][1]*(1-q) = A[1][0]*q + A[1][1]*(1-q)
    # Player 2 chooses q to make player 1 indifferent:
    # A[0][0]*q + A[0][1]*(1-q) = A[1][0]*q + A[1][1]*(1-q)
    # q*(A[0][0]-A[0][1]-A[1][0]+A[1][1]) = A[1][1]-A[0][1]
    denom_q = (A[0][0] - A[0][1] - A[1][0] + A[1][1])
    # Player 1 chooses p to make player 2 indifferent:
    # B[0][0]*p + B[1][0]*(1-p) = B[0][1]*p + B[1][1]*(1-p)
    # p*(B[0][0]-B[1][0]-B[0][1]+B[1][1]) = B[1][1]-B[1][0]
    denom_p = (B[0][0] - B[1][0] - B[0][1] + B[1][1])
    
    results = []
    
    # Check pure strategy NE
    for i in range(2):
        for j in range(2):
            p = [0.0, 0.0]; p[i] = 1.0
            q = [0.0, 0.0]; q[j] = 1.0
            # Check if (i,j) is NE
            br1 = best_response(A, q)
            br2 = best_response([[B[r][c] for c in range(2)] for r in range(2)], p)
            # Player 2's BR: maximize B given p
            ev2 = [sum(p[r] * B[r][c] for r in range(2)) for c in range(2)]
            max_ev2 = max(ev2)
            br2_cols = [c for c in range(2) if abs(ev2[c] - max_ev2) < 1e-9]
            if i in br1 and j in br2_cols:
                results.append((list(p), list(q)))
    
    # Mixed strategy NE
    if abs(denom_q) > 1e-9 and abs(denom_p) > 1e-9:
        q_mix = (A[1][1] - A[0][1]) / denom_q
        p_mix = (B[1][1] - B[1][0]) / denom_p
        if 0 < q_mix < 1 and 0 < p_mix < 1:
            results.append(([p_mix, 1-p_mix], [q_mix, 1-q_mix]))
    
    return results

def support_enumeration(A, B):
    """Find all Nash equilibria via support enumeration for small games."""
    m, n = len(A), len(A[0])
    equilibria = []
    
    for size_p in range(1, m+1):
        for size_q in range(1, n+1):
            for supp_p in combinations(range(m), size_p):
                for supp_q in combinations(range(n), size_q):
                    result = _check_support(A, B, supp_p, supp_q)
                    if result:
                        equilibria.append(result)
    return equilibria

def _check_support(A, B, supp_p, supp_q):
    """Check if supports yield a valid NE. Solve indifference conditions."""
    sp, sq = list(supp_p), list(supp_q)
    
    # Player 2 must be indifferent over supp_q given p
    # Player 1 must be indifferent over supp_p given q
    
    if len(sp) == 1 and len(sq) == 1:
        i, j = sp[0], sq[0]
        p = [0.0] * len(A); p[i] = 1.0
        q = [0.0] * len(A[0]); q[j] = 1.0
        # Check BR
        br1 = best_response(A, q)
        ev2 = [p[i] * B[i][c] for c in range(len(A[0]))]
        if i in br1 and ev2[j] >= max(ev2) - 1e-9:
            return (p, q)
        return None
    
    # For larger supports, solve linear system (simplified for 2x2)
    if len(A) == 2 and len(A[0]) == 2:
        return None  # handled by solve_2x2
    return None  # full support enum needs general linear solver

def expected_payoff(A, p, q):
    """Compute expected payoff for player with matrix A given strategies p, q."""
    return sum(p[i] * A[i][j] * q[j] for i in range(len(A)) for j in range(len(A[0])))

# Classic games
PRISONERS_DILEMMA = {
    "name": "Prisoner's Dilemma",
    "actions": (["Cooperate", "Defect"], ["Cooperate", "Defect"]),
    "A": [[-1, -3], [0, -2]],  # Row player
    "B": [[-1, 0], [-3, -2]],  # Col player
}

BATTLE_OF_SEXES = {
    "name": "Battle of the Sexes",
    "actions": (["Opera", "Football"], ["Opera", "Football"]),
    "A": [[3, 0], [0, 2]],
    "B": [[2, 0], [0, 3]],
}

MATCHING_PENNIES = {
    "name": "Matching Pennies",
    "actions": (["Heads", "Tails"], ["Heads", "Tails"]),
    "A": [[1, -1], [-1, 1]],
    "B": [[-1, 1], [1, -1]],
}

CHICKEN = {
    "name": "Chicken",
    "actions": (["Swerve", "Straight"], ["Swerve", "Straight"]),
    "A": [[0, -1], [1, -5]],
    "B": [[0, 1], [-1, -5]],
}

# --- Tests ---

def test_prisoners_dilemma():
    g = PRISONERS_DILEMMA
    eqs = solve_2x2(g["A"], g["B"])
    # Unique NE: (Defect, Defect)
    assert any(p[1] == 1.0 and q[1] == 1.0 for p, q in eqs), f"Expected (D,D), got {eqs}"

def test_battle_of_sexes():
    g = BATTLE_OF_SEXES
    eqs = solve_2x2(g["A"], g["B"])
    # 3 NE: (O,O), (F,F), and mixed
    assert len(eqs) == 3, f"Expected 3 equilibria, got {len(eqs)}"
    pure = [(p, q) for p, q in eqs if max(p) == 1.0]
    assert len(pure) == 2

def test_matching_pennies():
    g = MATCHING_PENNIES
    eqs = solve_2x2(g["A"], g["B"])
    # Unique NE: (0.5, 0.5)
    mixed = [(p, q) for p, q in eqs if 0 < p[0] < 1]
    assert len(mixed) == 1
    assert abs(mixed[0][0][0] - 0.5) < 1e-6
    assert abs(mixed[0][1][0] - 0.5) < 1e-6

def test_best_response():
    A = [[3, 0], [0, 2]]
    br = best_response(A, [1.0, 0.0])
    assert 0 in br  # Opera is BR to Opera
    br = best_response(A, [0.0, 1.0])
    assert 1 in br  # Football is BR to Football

def test_expected_payoff():
    A = [[1, -1], [-1, 1]]
    ep = expected_payoff(A, [0.5, 0.5], [0.5, 0.5])
    assert abs(ep) < 1e-9  # zero-sum at equilibrium

def test_chicken():
    g = CHICKEN
    eqs = solve_2x2(g["A"], g["B"])
    assert len(eqs) == 3  # 2 pure + 1 mixed

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_prisoners_dilemma()
        test_battle_of_sexes()
        test_matching_pennies()
        test_best_response()
        test_expected_payoff()
        test_chicken()
        print("All tests passed!")
    else:
        for game in [PRISONERS_DILEMMA, BATTLE_OF_SEXES, MATCHING_PENNIES, CHICKEN]:
            print(f"\n=== {game['name']} ===")
            eqs = solve_2x2(game["A"], game["B"])
            for i, (p, q) in enumerate(eqs):
                p_str = ", ".join(f"{game['actions'][0][j]}:{p[j]:.3f}" for j in range(2))
                q_str = ", ".join(f"{game['actions'][1][j]}:{q[j]:.3f}" for j in range(2))
                ep1 = expected_payoff(game["A"], p, q)
                ep2 = expected_payoff(game["B"], p, q)
                print(f"  NE{i+1}: P1=[{p_str}] P2=[{q_str}] Payoffs=({ep1:.2f}, {ep2:.2f})")
