from ortools.sat.python import cp_model

player_letters = "ABCDEFGHIJKLMNOPRSTUVWXYZ"

# Social Golfer Problem with additional constraints:
def solve(P=16, W=8, G=8, S=2):
    model = cp_model.CpModel()

    # x[p][w] = group of player p in week w (0..G-1)
    x = {}
    for p in range(P):
        for w in range(W):
            x[p, w] = model.NewIntVar(0, G - 1, f"x_p{p}_w{w}")

    # --------------------------------------------------
    # Each group has exactly 2 players per week
    for w in range(W):
        for g in range(G):
            bools = []
            for p in range(P):
                b = model.NewBoolVar(f"is_p{p}_w{w}_g{g}")
                model.Add(x[p, w] == g).OnlyEnforceIf(b)
                model.Add(x[p, w] != g).OnlyEnforceIf(b.Not())
                bools.append(b)
            model.Add(sum(bools) == S)

    # --------------------------------------------------
    # No player repeats same group index
    for p in range(P):
        model.AddAllDifferent([x[p, w] for w in range(W)])

    # --------------------------------------------------
    # No pair meets more than once
    for p1 in range(P):
        for p2 in range(p1 + 1, P):
            meet_bools = []
            for w in range(W):
                b = model.NewBoolVar(f"meet_p{p1}_{p2}_w{w}")
                model.Add(x[p1, w] == x[p2, w]).OnlyEnforceIf(b)
                model.Add(x[p1, w] != x[p2, w]).OnlyEnforceIf(b.Not())
                meet_bools.append(b)
            model.Add(sum(meet_bools) <= 1)

    # --------------------------------------------------
    # Symmetry breaking: fix week 0
    for g in range(G):
        model.Add(x[2*g,   0] == g)
        model.Add(x[2*g+1, 0] == g)

    # Strong symmetry break: player 0 visits groups in order
    for w in range(W):
        model.Add(x[0, w] == w)

    # --------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8

    result = solver.Solve(model)

    if result in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        rounds = {w: [] for w in range(W)}

        for w in range(W):
            groups = [[] for _ in range(G)]
            for p in range(P):
                g = solver.Value(x[p, w])
                groups[g].append(p + 1)
            rounds[w] = groups
        return rounds
    else:
        print("No solution found.")
        return {}

def print_schedule(schedule):
    print("|-------|" + "-----|" * len(schedule[0]))
    header = "| Round | " + " | ".join(f"St{g + 1}" for g in range(len(schedule[0]))) + " |"
    print(header)
    for w, groups in schedule.items():
        row = f"|   {w + 1}   | " + " | ".join(
            ",".join(player_letters[p - 1] for p in group) for group in groups
        ) + " |"
        print("|-------|" + "-----|" * len(schedule[0]))
        print(row)
    print("|-------|" + "-----|" * len(schedule[0]))

def verify_schedule(schedule):
    # Verify that no pair meets more than once
    pair_meetings = {}
    errors = []
    for w, groups in schedule.items():
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted((group[i], group[j])))
                    if pair in pair_meetings:
                        errors.append(f"Pair {pair} meets more than once: weeks {pair_meetings[pair]} and {w + 1}")
                    else:
                        pair_meetings[pair] = w + 1
    # Verify That each player is in exactly one group per week
    player_weeks = {p: set() for p in range(1, 17)}
    for w, groups in schedule.items():
        for group in groups:
            for p in group:
                if w + 1 in player_weeks[p]:
                    errors.append(f"Player {p} is in multiple groups in week {w + 1}")
                else:
                    player_weeks[p].add(w + 1)
    # Verify that each player visits each group index at most once
    player_groups = {p: set() for p in range(1, 17)}
    for w, groups in schedule.items():
        for g, group in enumerate(groups):
            for p in group:
                if g in player_groups[p]:
                    errors.append(f"Player {p} is in group {g} more than once (week {w + 1})")
                else:
                    player_groups[p].add(g)
    if errors:
        print("Schedule verification failed:")
        for error in errors:
            print(error)
    else:
        print("Schedule verification passed successfully.")

if __name__ == "__main__":
    players = 16
    if players % 2 != 0:
        print("Number of players must be even.")
        exit(1)
    rounds = players / 2
    stations = rounds
    group_size = 2

    schedule = solve(P=players, W=rounds, G=stations, S=group_size)
    print_schedule(schedule)
    verify_schedule(schedule)
