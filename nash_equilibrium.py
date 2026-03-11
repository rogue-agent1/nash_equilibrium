#!/usr/bin/env python3
"""Nash Equilibrium solver for 2-player normal form games."""
import sys

def solve_2x2(A, B):
    """Find mixed strategy Nash equilibrium for 2x2 game."""
    # Player 2 makes Player 1 indifferent
    d1 = (A[0][0] - A[1][0]) - (A[0][1] - A[1][1])
    q = (A[1][1] - A[1][0]) / d1 if abs(d1) > 1e-10 else 0.5
    # Player 1 makes Player 2 indifferent
    d2 = (B[0][0] - B[0][1]) - (B[1][0] - B[1][1])
    p = (B[1][1] - B[0][1]) / d2 if abs(d2) > 1e-10 else 0.5
    q, p = max(0, min(1, q)), max(0, min(1, p))
    # Expected payoffs
    eu1 = p*q*A[0][0] + p*(1-q)*A[0][1] + (1-p)*q*A[1][0] + (1-p)*(1-q)*A[1][1]
    eu2 = p*q*B[0][0] + p*(1-q)*B[0][1] + (1-p)*q*B[1][0] + (1-p)*(1-q)*B[1][1]
    return (p, 1-p), (q, 1-q), (eu1, eu2)

def find_pure_nash(A, B):
    """Find pure strategy Nash equilibria."""
    rows, cols = len(A), len(A[0]); equilibria = []
    for i in range(rows):
        for j in range(cols):
            if A[i][j] == max(A[r][j] for r in range(rows)) and \
               B[i][j] == max(B[i][c] for c in range(cols)):
                equilibria.append((i, j, A[i][j], B[i][j]))
    return equilibria

if __name__ == "__main__":
    # Prisoner's Dilemma
    A = [[-1,-3],[0,-2]]; B = [[-1,0],[-3,-2]]
    print("Prisoner's Dilemma:")
    pure = find_pure_nash(A, B)
    for i,j,a,b in pure: print(f"  Pure NE: ({i},{j}) payoffs=({a},{b})")
    # Battle of the Sexes
    A2 = [[3,0],[0,2]]; B2 = [[2,0],[0,3]]
    print("\nBattle of the Sexes:")
    for i,j,a,b in find_pure_nash(A2, B2): print(f"  Pure NE: ({i},{j}) payoffs=({a},{b})")
    p, q, eu = solve_2x2(A2, B2)
    print(f"  Mixed NE: P1={p}, P2={q}, EU={eu}")
    # Matching Pennies (no pure NE)
    A3 = [[1,-1],[-1,1]]; B3 = [[-1,1],[1,-1]]
    print("\nMatching Pennies:")
    print(f"  Pure NE: {find_pure_nash(A3, B3)}")
    p, q, eu = solve_2x2(A3, B3)
    print(f"  Mixed NE: P1={p}, P2={q}, EU={eu}")
