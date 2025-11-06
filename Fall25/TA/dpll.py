def dpll(clauses, assignment={}):
    """
    DPLL algorithm.
    clauses: list of lists, each inner list is a clause of literals (ints)
    assignment: dict of {var: bool}
    """

    # Remove satisfied clauses
    clauses = [c for c in clauses if not any(lit in assignment and assignment[abs(lit)] == (lit > 0) for lit in c)]

    # Empty clause means unsatisfiable
    if any(len(c) == 0 for c in clauses):
        return None

    # No clauses left -> satisfiable
    if not clauses:
        return assignment

    # Unit propagation
    for clause in clauses:
        if len(clause) == 1:
            lit = clause[0]
            assignment[abs(lit)] = lit > 0
            return dpll(clauses, assignment)

    # Pure literal elimination
    all_lits = {lit for clause in clauses for lit in clause}
    for lit in list(all_lits):
        if -lit not in all_lits:
            assignment[abs(lit)] = lit > 0
            clauses = [c for c in clauses if lit not in c]
            return dpll(clauses, assignment)

    # Choose variable
    lit = clauses[0][0]
    for val in [True, False]:
        local_assign = assignment.copy()
        local_assign[abs(lit)] = val
        res = dpll(clauses, local_assign)
        if res is not None:
            return res
    return None


def parse_dimacs(path):
    """Simple CNF parser for DIMACS format"""
    clauses = []
    with open(path) as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            lits = [int(x) for x in line.strip().split() if x != '0']
            if lits:
                clauses.append(lits)
    return clauses


if __name__ == "__main__":
    cnf = parse_dimacs("map-coloring.cnf")
    sol = dpll(cnf)
    print("SATISFIABLE" if sol else "UNSATISFIABLE")
    if sol:
        print(sol)
