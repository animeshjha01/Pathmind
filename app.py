from flask import Flask, render_template, request, jsonify
from collections import deque
import heapq, time, sys

app = Flask(__name__)
sys.setrecursionlimit(5000)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_neighbors(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            yield (nr, nc)


def reconstruct(parent, start, end):
    if end not in parent:
        return []
    path, cur = [], end
    while cur is not None:
        path.append(list(cur))
        cur = parent[cur]
    path.reverse()
    return path if path and path[0] == list(start) else []


# ── Uninformed Search ──────────────────────────────────────────────────────────

def bfs(grid, s, e):
    """Breadth-First Search — Queue (FIFO). Optimal for unweighted graphs."""
    queue = deque([s])
    parent = {s: None}
    order = []
    while queue:
        u = queue.popleft()
        order.append(list(u))
        if u == e:
            break
        for v in get_neighbors(grid, *u):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return order, reconstruct(parent, s, e)


def dfs(grid, s, e):
    """Depth-First Search — Stack (LIFO). Not guaranteed optimal."""
    stack = [s]
    parent = {s: None}
    order = []
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(list(u))
        if u == e:
            break
        for v in get_neighbors(grid, *u):
            if v not in parent:
                parent[v] = u
                stack.append(v)
    return order, reconstruct(parent, s, e)


def ucs(grid, s, e):
    """Uniform Cost Search — Priority Queue (min-heap). Dijkstra on grid."""
    heap = [(0, s)]
    dist = {s: 0}
    parent = {s: None}
    order = []
    seen = set()
    while heap:
        cost, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        order.append(list(u))
        if u == e:
            break
        for v in get_neighbors(grid, *u):
            step_cost = grid[v[0]][v[1]] if grid[v[0]][v[1]] > 1 else 1
            nc = cost + step_cost
            if v not in dist or nc < dist[v]:
                dist[v] = nc
                parent[v] = u
                heapq.heappush(heap, (nc, v))
    return order, reconstruct(parent, s, e)


def dls(grid, s, e, limit):
    """Depth-Limited Search — DFS with a hard depth cutoff."""
    parent = {s: None}
    order = []
    seen = set()

    def recurse(u, depth):
        if depth > limit:
            return False
        seen.add(u)
        order.append(list(u))
        if u == e:
            return True
        for v in get_neighbors(grid, *u):
            if v not in seen:
                parent[v] = u
                if recurse(v, depth + 1):
                    return True
        return False

    recurse(s, 0)
    return order, reconstruct(parent, s, e)


# ── Informed Search ────────────────────────────────────────────────────────────

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, s, e):
    """A* Search — Priority Queue with f(n) = g(n) + h(n). Optimal."""
    heap = [(manhattan(s, e), 0, s)]
    g = {s: 0}
    parent = {s: None}
    order = []
    seen = set()
    while heap:
        _, cost, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        order.append(list(u))
        if u == e:
            break
        for v in get_neighbors(grid, *u):
            step_cost = grid[v[0]][v[1]] if grid[v[0]][v[1]] > 1 else 1
            nc = cost + step_cost
            if v not in g or nc < g[v]:
                g[v] = nc
                parent[v] = u
                heapq.heappush(heap, (nc + manhattan(v, e), nc, v))
    return order, reconstruct(parent, s, e)


def greedy(grid, s, e):
    """Greedy Best-First — Priority Queue using h(n) only. Not optimal."""
    heap = [(manhattan(s, e), s)]
    parent = {s: None}
    order = []
    seen = set()
    while heap:
        _, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        order.append(list(u))
        if u == e:
            break
        for v in get_neighbors(grid, *u):
            if v not in parent:
                parent[v] = u
                heapq.heappush(heap, (manhattan(v, e), v))
    return order, reconstruct(parent, s, e)


# ── Routes ─────────────────────────────────────────────────────────────────────

ALGOS = {
    'bfs': bfs,
    'dfs': dfs,
    'ucs': ucs,
    'astar': astar,
    'greedy': greedy,
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    grid   = data['grid']
    start  = tuple(data['start'])
    end    = tuple(data['end'])
    algo   = data['algorithm']
    limit  = int(data.get('limit', 25))

    t0 = time.perf_counter()

    try:
        if algo == 'dls':
            visited, path = dls(grid, start, end, limit)
        elif algo in ALGOS:
            visited, path = ALGOS[algo](grid, start, end)
        else:
            return jsonify({'error': f'Unknown algorithm: {algo}'}), 400
    except RecursionError:
        return jsonify({'error': 'Recursion limit hit — reduce depth limit'}), 400

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        'visited': visited,
        'path': path,
        'stats': {
            'nodesExplored': len(visited),
            'pathLength':    len(path) - 1 if path else 0,
            'timeMs':        round(elapsed_ms, 3),
            'found':         len(path) > 0,
            'optimal':       algo in ('bfs', 'ucs', 'astar'),
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)