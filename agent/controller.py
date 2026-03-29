from collections.abc import Callable, Iterable
from collections import deque
from typing import Union
import random
import math
import time as time_module

from game import *


# Direction offsets: UP(-1,0), DOWN(1,0), LEFT(0,-1), RIGHT(0,1)
DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]


class LightBoard:
    """
    Lightweight board representation for forward simulation.
    Uses plain Python lists — no numpy. Only tracks what's needed
    to evaluate territorial outcomes.
    """

    __slots__ = (
        'rows', 'cols', 'paint', 'walls', 'hill_ids', 'hill_set',
        'my_r', 'my_c', 'opp_r', 'opp_c',
        'my_stamina', 'opp_stamina',
        'my_territory', 'opp_territory',
    )

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.paint = None  # 2D list: 1 = me, -1 = opp, 0 = neutral
        self.walls = None  # 2D list of bool
        self.hill_ids = None  # 2D list of int (0 = not hill)
        self.hill_set = set()  # set of (r, c) that are hill cells
        self.my_r = 0
        self.my_c = 0
        self.opp_r = 0
        self.opp_c = 0
        self.my_stamina = 100
        self.opp_stamina = 100
        self.my_territory = 0
        self.opp_territory = 0

    @staticmethod
    def from_game_board(board, parity, me, opp):
        """Create a LightBoard from the real game Board."""
        rows = len(board.cells)
        cols = len(board.cells[0])
        lb = LightBoard(rows, cols)

        # Build paint and walls arrays
        paint = []
        walls = []
        hill_ids = []
        my_t = 0
        opp_t = 0
        for r in range(rows):
            paint_row = []
            wall_row = []
            hill_row = []
            for c in range(cols):
                cell = board.cells[r][c]
                if cell.is_wall:
                    wall_row.append(True)
                    paint_row.append(0)
                    hill_row.append(0)
                else:
                    wall_row.append(False)
                    op = cell.owner_parity
                    if op == parity:
                        paint_row.append(1)
                        my_t += 1
                    elif op == -parity:
                        paint_row.append(-1)
                        opp_t += 1
                    else:
                        paint_row.append(0)
                    hid = cell.hill_id if cell.hill_id else 0
                    hill_row.append(hid)
                    if hid != 0:
                        lb.hill_set.add((r, c))
            paint.append(paint_row)
            walls.append(wall_row)
            hill_ids.append(hill_row)

        lb.paint = paint
        lb.walls = walls
        lb.hill_ids = hill_ids
        lb.my_territory = my_t
        lb.opp_territory = opp_t

        lb.my_r = me.loc.r
        lb.my_c = me.loc.c
        lb.opp_r = opp.loc.r if opp else -100
        lb.opp_c = opp.loc.c if opp else -100
        lb.my_stamina = me.stamina
        lb.opp_stamina = opp.stamina if opp else 0

        return lb

    def copy(self):
        """Fast shallow copy — copies paint grid, shares walls/hills."""
        lb = LightBoard(self.rows, self.cols)
        # Deep copy paint (it mutates); share walls/hills (they don't)
        lb.paint = [row[:] for row in self.paint]
        lb.walls = self.walls
        lb.hill_ids = self.hill_ids
        lb.hill_set = self.hill_set
        lb.my_r = self.my_r
        lb.my_c = self.my_c
        lb.opp_r = self.opp_r
        lb.opp_c = self.opp_c
        lb.my_stamina = self.my_stamina
        lb.opp_stamina = self.opp_stamina
        lb.my_territory = self.my_territory
        lb.opp_territory = self.opp_territory
        return lb

    def _valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and not self.walls[r][c]

    def _paint_cell(self, r, c, owner):
        """Paint a cell (owner: 1=me, -1=opp). Returns True if changed."""
        if not self._valid(r, c):
            return False
        old = self.paint[r][c]
        if old == owner:
            return False
        # Can only paint neutral or own cells (no enemy overwrite without erase)
        if old != 0 and old != owner:
            return False
        self.paint[r][c] = owner
        if owner == 1:
            self.my_territory += 1
        elif owner == -1:
            self.opp_territory += 1
        return True

    def _best_direction_for(self, r, c, opp_r, opp_c, owner, depth=3):
        """
        Simple BFS scoring to pick best direction for either side.
        Scores unpainted/enemy cells within depth, returns best (di index, score).
        """
        enemy = -owner
        best_di = -1
        best_score = -1

        for di in range(4):
            nr, nc = r + DR[di], c + DC[di]
            if not self._valid(nr, nc):
                continue

            score = 0.0
            # Immediate cell value
            p = self.paint[nr][nc]
            if p == 0:
                score += 2.0
            elif p == enemy:
                score += 1.5

            if (nr, nc) in self.hill_set and p != owner:
                score += 15.0

            # BFS deeper
            visited = {(r, c), (nr, nc)}
            frontier = [(nr, nc)]
            weights = [1.5, 1.0, 0.5]
            for d in range(min(depth - 1, len(weights))):
                w = weights[d]
                next_f = []
                for fr, fc in frontier:
                    for di2 in range(4):
                        nr2, nc2 = fr + DR[di2], fc + DC[di2]
                        if not self._valid(nr2, nc2):
                            continue
                        if (nr2, nc2) in visited:
                            continue
                        visited.add((nr2, nc2))
                        p2 = self.paint[nr2][nc2]
                        if p2 == 0:
                            score += w
                        elif p2 == enemy:
                            score += w * 0.7
                        if (nr2, nc2) in self.hill_set and p2 != owner:
                            score += 15.0
                        next_f.append((nr2, nc2))
                    frontier = next_f

            # Tiebreaker: away from opponent
            if opp_r >= 0:
                dist = abs(nr - opp_r) + abs(nc - opp_c)
                score += dist * 0.08

            if score > best_score:
                best_score = score
                best_di = di

        return best_di, best_score

    def _apply_regen_for(self, owner):
        """Approximate stamina regen: base 5 + adj*2 + territory/8, capped at 100."""
        if owner == 1:
            r, c = self.my_r, self.my_c
            territory = self.my_territory
            stamina = self.my_stamina
        else:
            r, c = self.opp_r, self.opp_c
            territory = self.opp_territory
            stamina = self.opp_stamina

        adj_count = 0
        for di in range(4):
            nr, nc = r + DR[di], c + DC[di]
            if self._valid(nr, nc) and self.paint[nr][nc] == owner:
                adj_count += 1
        regen = 5 + adj_count * 2 + territory // 8
        if regen > 50:
            regen = 50
        stamina = min(100, stamina + regen)
        if owner == 1:
            self.my_stamina = stamina
        else:
            self.opp_stamina = stamina

    def _simulate_side_turn(self, owner):
        """
        Simulate one simplified turn for either side.
        """
        if owner == 1:
            r, c = self.my_r, self.my_c
            opp_r, opp_c = self.opp_r, self.opp_c
            stamina = self.my_stamina
        else:
            r, c = self.opp_r, self.opp_c
            opp_r, opp_c = self.my_r, self.my_c
            stamina = self.opp_stamina

        # Paint adjacent neutral cells from current position
        for di in range(4):
            nr, nc = r + DR[di], c + DC[di]
            if self._valid(nr, nc) and self.paint[nr][nc] == 0:
                if stamina >= 30:
                    self._paint_cell(nr, nc, owner)
                    stamina -= 15

        # Find best direction
        best_di, _ = self._best_direction_for(r, c, opp_r, opp_c, owner, depth=3)
        if best_di < 0:
            if owner == 1:
                self.my_stamina = stamina
            else:
                self.opp_stamina = stamina
            self._apply_regen_for(owner)
            return

        # Move
        nr, nc = r + DR[best_di], c + DC[best_di]
        if owner == 1:
            self.my_r, self.my_c = nr, nc
        else:
            self.opp_r, self.opp_c = nr, nc

        # Paint from new position
        for di in range(4):
            pr, pc = nr + DR[di], nc + DC[di]
            if self._valid(pr, pc) and self.paint[pr][pc] == 0:
                if stamina >= 15:
                    self._paint_cell(pr, pc, owner)
                    stamina -= 15

        # Multi-move if stamina allows
        if stamina >= 25:
            best_di2, _ = self._best_direction_for(nr, nc, opp_r, opp_c, owner, depth=2)
            if best_di2 >= 0:
                nr2, nc2 = nr + DR[best_di2], nc + DC[best_di2]
                if self._valid(nr2, nc2):
                    # Safety: don't step on enemy cell near opponent
                    opp_dist = abs(nr2 - opp_r) + abs(nc2 - opp_c) if opp_r >= 0 else 999
                    if not (opp_dist <= 4 and self.paint[nr2][nc2] == -owner):
                        if owner == 1:
                            self.my_r, self.my_c = nr2, nc2
                        else:
                            self.opp_r, self.opp_c = nr2, nc2
                        stamina -= 10

                        # Paint from multi-move position
                        for di in range(4):
                            pr, pc = nr2 + DR[di], nc2 + DC[di]
                            if self._valid(pr, pc) and self.paint[pr][pc] == 0:
                                if stamina >= 15:
                                    self._paint_cell(pr, pc, owner)
                                    stamina -= 15

        if owner == 1:
            self.my_stamina = stamina
        else:
            self.opp_stamina = stamina
        self._apply_regen_for(owner)

    def simulate_storm_turn(self):
        """
        Simulate one simplified future round: opponent turn, then our turn.
        The first candidate move has already spent our current turn.
        """
        self._simulate_side_turn(-1)
        self._simulate_side_turn(1)

    def evaluate(self):
        """
        Score the position from 'my' perspective.
        territory diff * 2.0 + hill control * 100.0
        """
        my_hills = 0
        opp_hills = 0
        for (hr, hc) in self.hill_set:
            p = self.paint[hr][hc]
            if p == 1:
                my_hills += 1
            elif p == -1:
                opp_hills += 1

        score = (self.my_territory - self.opp_territory) * 2.0
        score += my_hills * 100.0 - opp_hills * 100.0
        return score


class PlayerController:
    """
    mcts_v6_f: Heavier simulation: always 5 turns ahead, 4 candidates
    Uses a LightBoard to simulate 3-5 turns ahead for top candidate
    directions, picking the one with best territorial outcome.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.turn = 0
        self.hill_cells = None
        self.hill_set = None
        self.last_opp_dist = 999
        self.opp_closing_turns = 0
        self.map_tier = 'large'
        self.territory_behind = False
        self.safe_dist = 8
        self.center_r = 0
        self.center_c = 0
        self.opp_history = []
        self._path_dist_to_opp = 999
        self.bfs_depth = 5  # default, adjusted per map size
        self.my_history = []

    def _discover_hills(self, board, rows, cols):
        if self.hill_cells is not None:
            return
        self.hill_cells = {}
        self.hill_set = set()
        for r in range(rows):
            for c in range(cols):
                cell = board.cells[r][c]
                hid = cell.hill_id
                if hid and hid != 0:
                    if hid not in self.hill_cells:
                        self.hill_cells[hid] = []
                    self.hill_cells[hid].append((r, c))
                    self.hill_set.add((r, c))
        if not self.hill_cells:
            try:
                hm = board.hill_mapping
                for r in range(rows):
                    for c in range(cols):
                        v = hm[r][c]
                        if v != 0:
                            if v not in self.hill_cells:
                                self.hill_cells[v] = []
                            self.hill_cells[v].append((r, c))
                            self.hill_set.add((r, c))
            except Exception:
                pass

    def _find_target_hill(self, board, me, parity, opp_r=-100, opp_c=-100):
        if not self.hill_cells:
            return set()

        # Count controlled hills for DOMINATION prevention
        my_hills = 0
        opp_hills = 0
        for hid, cells in self.hill_cells.items():
            total = len(cells)
            threshold = math.ceil(total * GameConstants.HILL_CONTROL_THRESHOLD)
            my = sum(1 for r, c in cells if board.cells[r][c].owner_parity == parity)
            opp = sum(1 for r, c in cells if board.cells[r][c].owner_parity == -parity)
            if my >= threshold and my > opp:
                my_hills += 1
            elif opp >= threshold and opp > my:
                opp_hills += 1
        panic = opp_hills >= my_hills  # Panic on ties too

        targets = []
        for hid, cells in self.hill_cells.items():
            my = sum(1 for r, c in cells if board.cells[r][c].owner_parity == parity)
            opp = sum(1 for r, c in cells if board.cells[r][c].owner_parity == -parity)
            total = len(cells)
            threshold = math.ceil(total * GameConstants.HILL_CONTROL_THRESHOLD)
            if panic:
                if opp >= my or my < threshold:
                    min_d_me = min(abs(me.loc.r - r) + abs(me.loc.c - c) for r, c in cells)
                    own_factor = 2.0 * (my - opp) / total if total > 0 else 0
                    targets.append((min_d_me + own_factor, hid))
            else:
                needs_target = my <= opp or my < threshold
                # Reinforce narrowly-held hills
                if not needs_target and opp > 0 and (my - opp) <= 2:
                    needs_target = True
                # PROACTIVE: also target hills we haven't captured yet (pursue domination)
                if not needs_target and my < threshold:
                    needs_target = True
                if needs_target:
                    min_d_me = min(abs(me.loc.r - r) + abs(me.loc.c - c) for r, c in cells)
                    own_factor = 2.0 * (my - opp) / total if total > 0 else 0
                    if opp_r >= 0:
                        min_d_opp = min(abs(opp_r - r) + abs(opp_c - c) for r, c in cells)
                        score = min_d_me - 0.5 * min_d_opp + own_factor
                    else:
                        score = min_d_me + own_factor
                    targets.append((score, hid))
        if not targets:
            return set()
        targets.sort()
        return set(self.hill_cells[targets[0][1]])

    def _is_opponent_rushing(self, me_r, me_c):
        """Check if opponent has been consistently moving toward us over last 3 turns."""
        if len(self.opp_history) < 3:
            return False
        # Check if each successive position is closer to us
        dists = []
        for (r, c) in self.opp_history[-3:]:
            dists.append(abs(r - me_r) + abs(c - me_c))
        # Opponent is rushing if distance has been strictly decreasing
        return dists[0] > dists[1] > dists[2]

    def _bfs_path_dist(self, board, sr, sc, tr, tc, max_dist=20):
        """BFS path distance from (sr,sc) to (tr,tc), respecting walls. Returns max_dist if unreachable."""
        if sr == tr and sc == tc:
            return 0
        visited = {(sr, sc)}
        frontier = [(sr, sc)]
        dist = 0
        while frontier and dist < max_dist:
            dist += 1
            next_f = []
            for fr, fc in frontier:
                for d in Direction.cardinals():
                    nl = Location(fr, fc) + d
                    if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                        continue
                    if (nl.r, nl.c) in visited:
                        continue
                    if nl.r == tr and nl.c == tc:
                        return dist
                    visited.add((nl.r, nl.c))
                    next_f.append((nl.r, nl.c))
            frontier = next_f
        return max_dist

    def _build_danger_zone(self, board, opp):
        """Build danger zone with radius 3 (opponent can multi-move 3 cells/turn)."""
        danger = set()
        if not opp:
            return danger
        opp_r, opp_c = opp.loc.r, opp.loc.c
        danger.add((opp_r, opp_c))
        # Radius 1
        frontier = [(opp_r, opp_c)]
        visited = {(opp_r, opp_c)}
        radius = 1 if self.map_tier == 'small' else 2
        for step in range(radius):
            next_frontier = []
            for fr, fc in frontier:
                for d in Direction.cardinals():
                    nl = Location(fr, fc) + d
                    if not board.oob(nl) and (nl.r, nl.c) not in visited:
                        visited.add((nl.r, nl.c))
                        danger.add((nl.r, nl.c))
                        next_frontier.append((nl.r, nl.c))
            frontier = next_frontier
        return danger

    def _max_regular_moves(self, stamina_budget):
        moves = 1
        next_cost = GameConstants.EXTRA_MOVE_COST
        while stamina_budget >= next_cost:
            stamina_budget -= next_cost
            moves += 1
            next_cost += GameConstants.EXTRA_MOVE_COST
        return moves

    def _shortest_path_dirs(self, board, start_r, start_c, goal_r, goal_c, max_depth):
        queue = deque([(start_r, start_c, 0)])
        parents = {(start_r, start_c): None}

        while queue:
            r, c, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for d in Direction.cardinals():
                nl = Location(r, c) + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in parents:
                    continue
                parents[(nl.r, nl.c)] = ((r, c), d)
                if nl.r == goal_r and nl.c == goal_c:
                    path = []
                    cur = (nl.r, nl.c)
                    while parents[cur] is not None:
                        prev, move_dir = parents[cur]
                        path.append(move_dir)
                        cur = prev
                    path.reverse()
                    return path
                queue.append((nl.r, nl.c, depth + 1))
        return None

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        try:
            me = board.get_player(player_parity)
            opp = board.get_player(-player_parity)
            dist = abs(me.loc.r - opp.loc.r) + abs(me.loc.c - opp.loc.c)
            # Turn 1: adaptive bid based on map size
            if self.turn == 0:
                rows = len(board.cells)
                cols = len(board.cells[0])
                cells = rows * cols
                if cells < 225:
                    return 12
                elif cells <= 500:
                    return 8
                else:
                    return 3
            # Only bid high for chase: close AND opponent on our territory
            if dist <= 4:
                opp_cell = board.cells[opp.loc.r][opp.loc.c]
                if opp_cell.owner_parity == player_parity:
                    return 15
            return 0
        except Exception:
            return 0

    def _find_move_candidates(self, board, me, parity, rows, cols,
                              opp_r, opp_c, danger, near_opp, target,
                              effective_safe_dist=None, powerup_dir=None):
        """
        Modified _find_move that returns a list of (score, direction) tuples
        for the top candidates, instead of just the best direction.
        """
        if effective_safe_dist is None:
            effective_safe_dist = self.safe_dist
        start = me.loc

        dist_to_opp = self._path_dist_to_opp

        # If we have a target hill, use direct BFS to find it first
        if target:
            direct = self._bfs_to_target(board, start, parity, rows, cols,
                                         opp_r, opp_c, danger, near_opp, target,
                                         dist_to_opp, effective_safe_dist, me)
            if direct:
                # Target hill BFS gives a single strong direction; return it as sole candidate
                return [(999.0, direct)]

        # SCORED BFS: for each valid initial direction, count unpainted cells within depth
        dirs = list(Direction.cardinals())
        candidates = []

        for d in dirs:
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            # Only avoid stepping on opponent's location if it's THEIR cell
            if nl.r == opp_r and nl.c == opp_c and cell.owner_parity == -parity:
                continue
            # Near opp: only avoid ENEMY cells in danger zone (neutral is safe for us)
            if near_opp and (nl.r, nl.c) in danger and cell.owner_parity == -parity:
                continue
            if dist_to_opp <= effective_safe_dist and cell.owner_parity == -parity:
                continue

            # BFS from this neighbor up to bfs_depth, distance-weighted scoring
            score = 0
            visited = {(start.r, start.c), (nl.r, nl.c)}
            frontier = [(nl.r, nl.c)]
            # Steep weights: heavily favor immediate cells
            steep_weights = [2.0, 1.5, 1.0, 0.5, 0.3]
            weight = steep_weights[0]
            if cell.owner_parity == 0:
                score += weight
                # Frontier bonus: neutral cell next to our territory
                for d3 in Direction.cardinals():
                    adj = nl + d3
                    if not board.oob(adj) and board.cells[adj.r][adj.c].owner_parity == parity:
                        score += 0.4
                        break
            elif cell.owner_parity == -parity:
                score += weight * (0.9 if self.territory_behind else 0.7)
            if (nl.r, nl.c) in self.hill_set and cell.owner_parity != parity:
                score += 15

            # Anti-backtrack
            if (nl.r, nl.c) in set(self.my_history[-4:]):
                score -= 0.5

            for depth in range(self.bfs_depth - 1):  # adaptive depth
                weight = steep_weights[min(depth + 1, len(steep_weights) - 1)]
                next_frontier = []
                for fr, fc in frontier:
                    for d2 in Direction.cardinals():
                        nl2 = Location(fr, fc) + d2
                        if board.oob(nl2) or board.cells[nl2.r][nl2.c].is_wall:
                            continue
                        if (nl2.r, nl2.c) in visited:
                            continue
                        visited.add((nl2.r, nl2.c))
                        c2 = board.cells[nl2.r][nl2.c]
                        if c2.owner_parity == 0:
                            # Bonus for unpainted cells adjacent to our territory
                            adj_bonus = 0
                            for d3 in Direction.cardinals():
                                adj = Location(nl2.r, nl2.c) + d3
                                if not board.oob(adj) and board.cells[adj.r][adj.c].owner_parity == parity:
                                    adj_bonus += 0.25
                            score += weight + adj_bonus
                        elif c2.owner_parity == -parity:
                            score += weight * (0.9 if self.territory_behind else 0.7)
                        if (nl2.r, nl2.c) in self.hill_set and c2.owner_parity != parity:
                            score += 15
                        next_frontier.append((nl2.r, nl2.c))
                frontier = next_frontier

            # Add small tiebreaker favoring moves away from opponent
            if opp_r >= 0:
                nl_dist = abs(nl.r - opp_r) + abs(nl.c - opp_c)
                score += nl_dist * 0.10

            # Powerup direction bias
            if powerup_dir and d == powerup_dir:
                score += 0.5

            candidates.append((score, d))

        if not candidates:
            # Fallback: any unpainted via regular BFS
            fallback = self._bfs_unpainted(board, start, parity, rows, cols,
                                           opp_r, opp_c, danger, near_opp,
                                           effective_safe_dist)
            if fallback:
                return [(0.0, fallback)]

            # All painted: seek enemy border
            expand = self._find_enemy_border(board, start, parity, rows, cols, danger, near_opp,
                                             opp_r, opp_c, effective_safe_dist)
            if expand:
                return [(0.0, expand)]

            # Last resort: chase on owned territory
            if opp_r >= 0:
                chase = self._bfs_toward(board, start, Location(opp_r, opp_c), rows, cols)
                if chase:
                    s = start + chase
                    if not board.oob(s) and board.cells[s.r][s.c].owner_parity == parity:
                        return [(0.0, chase)]
            return []

        # Sort descending by score, return top candidates
        candidates.sort(key=lambda x: -x[0])
        return candidates

    def _simulate_candidates(self, board, me, parity, candidates, opp,
                             sim_turns, max_candidates):
        """
        For each candidate direction, simulate sim_turns forward using LightBoard
        and return the direction with the best evaluated outcome.
        """
        if not candidates:
            return None

        # Limit number of candidates to evaluate
        to_eval = candidates[:max_candidates]

        best_dir = to_eval[0][1]  # default: highest scoring from BFS
        best_eval = -999999.0

        for _, cand_dir in to_eval:
            # Create LightBoard snapshot
            lb = LightBoard.from_game_board(board, parity, me, opp)

            # Apply the candidate's first move direction
            dr_val = cand_dir.value[0]
            dc_val = cand_dir.value[1]
            nr = lb.my_r + dr_val
            nc = lb.my_c + dc_val
            if not lb._valid(nr, nc):
                continue

            # Move to the candidate cell
            lb.my_r = nr
            lb.my_c = nc

            # Paint from new position (simulate the initial paint actions)
            stamina = lb.my_stamina
            for di in range(4):
                pr, pc = nr + DR[di], nc + DC[di]
                if lb._valid(pr, pc) and lb.paint[pr][pc] == 0:
                    if stamina >= 15:
                        lb._paint_cell(pr, pc, 1)
                        stamina -= 15
            lb.my_stamina = stamina
            lb._apply_regen_for(1)

            # Simulate subsequent turns
            for _ in range(sim_turns - 1):
                lb.simulate_storm_turn()

            # Evaluate
            ev = lb.evaluate()
            if ev > best_eval:
                best_eval = ev
                best_dir = cand_dir

        return best_dir

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        self.turn += 1
        me = board.get_player(player_parity)
        rows = len(board.cells)
        cols = len(board.cells[0])
        self.my_history.append((me.loc.r, me.loc.c))
        if len(self.my_history) > 6:
            self.my_history.pop(0)
        if self.turn == 1:
            cells = rows * cols
            if cells < 225:
                self.map_tier = 'small'
                self.safe_dist = 4
                self.bfs_depth = 3
            elif cells <= 500:
                self.map_tier = 'medium'
                self.safe_dist = 4
                self.bfs_depth = 4
            else:
                self.map_tier = 'large'
                wall_count = sum(1 for r in range(rows) for c in range(cols) if board.cells[r][c].is_wall)
                wall_ratio = wall_count / cells
                self.safe_dist = 4
                self.bfs_depth = 6
        self._discover_hills(board, rows, cols)
        if self.turn == 1:
            self.center_r = rows // 2
            self.center_c = cols // 2

        # Territory counting (every 20 turns to save compute)
        if self.turn % 20 == 0 and self.turn >= 100:
            my_t = 0
            opp_t = 0
            for r in range(rows):
                for c in range(cols):
                    op = board.cells[r][c].owner_parity
                    if op == player_parity:
                        my_t += 1
                    elif op == -player_parity:
                        opp_t += 1
            self.territory_behind = opp_t > my_t

        try:
            opp = board.get_player(-player_parity)
            opp_r, opp_c = opp.loc.r, opp.loc.c
        except Exception:
            opp = None
            opp_r, opp_c = -100, -100

        if opp and board.cells[opp_r][opp_c].owner_parity != -player_parity:
            effective_stamina = me.stamina
            if board.cells[me.loc.r][me.loc.c].powerup:
                effective_stamina = min(
                    me.max_stamina,
                    me.stamina + GameConstants.STAMINA_POWERUP_AMOUNT,
                )
            kill_path = self._shortest_path_dirs(
                board,
                me.loc.r,
                me.loc.c,
                opp_r,
                opp_c,
                self._max_regular_moves(effective_stamina),
            )
            if kill_path:
                return [Action.Move(move_dir) for move_dir in kill_path]

        # Track opponent position history for velocity
        if opp:
            self.opp_history.append((opp_r, opp_c))
            if len(self.opp_history) > 5:
                self.opp_history.pop(0)

        manhattan_dist = abs(me.loc.r - opp_r) + abs(me.loc.c - opp_c)
        # Use BFS path distance for more accurate collision avoidance
        if opp and manhattan_dist <= 15:
            dist_to_opp = self._bfs_path_dist(board, me.loc.r, me.loc.c, opp_r, opp_c, 20)
        else:
            dist_to_opp = manhattan_dist
        # Store for use by sub-functions
        self._path_dist_to_opp = dist_to_opp

        # APPROACH DETECTION: opponent closing in fast
        approach = self.last_opp_dist - dist_to_opp
        if dist_to_opp < self.last_opp_dist:
            self.opp_closing_turns += 1
        else:
            self.opp_closing_turns = 0
        self.last_opp_dist = dist_to_opp
        approaching = approach >= 2  # opponent moved 2+ cells closer

        # VELOCITY-BASED safe_dist boost
        effective_safe_dist = self.safe_dist
        if self._is_opponent_rushing(me.loc.r, me.loc.c):
            effective_safe_dist += 2

        local = self._count_local(board, me.loc, player_parity, rows, cols)

        # Build expanded danger zone (radius 3)
        danger = self._build_danger_zone(board, opp)

        # Build chase buffer zone: danger zone + 1 cell radius
        chase_danger = set(danger)
        if opp:
            for (dr, dc) in list(danger):
                for d in Direction.cardinals():
                    nl = Location(dr, dc) + d
                    if not board.oob(nl):
                        chase_danger.add((nl.r, nl.c))

        # Simplified near_opp: just distance-based, no sustained approach or rushing triggers
        if self.map_tier == 'small':
            near_opp = dist_to_opp <= 4 or (approaching and dist_to_opp <= 6)
        elif self.map_tier == 'medium':
            near_opp = dist_to_opp <= 5 or (approaching and dist_to_opp <= 7)
        else:
            near_opp = dist_to_opp <= 4 or (approaching and dist_to_opp <= 6)

        # === POWERUP COLLECTION ===
        powerup_action = self._check_powerup(board, me, player_parity, danger, near_opp)
        if powerup_action:
            return powerup_action

        powerup_dir = self._bfs_powerup(board, me, player_parity, danger, near_opp, max_dist=4)
        if powerup_dir is None and self.map_tier == 'large' and not near_opp:
            powerup_dir = self._bfs_scheduled_powerup(
                board,
                me,
                player_parity,
                danger,
                near_opp,
                max_rounds=2,
                max_dist=6,
            )

        # === AGGRESSIVE COLLISION HUNTING ===
        if opp and dist_to_opp <= 5:
            opp_cell = board.cells[opp_r][opp_c]
            # Chase on OUR territory (we own the cell, we win collision)
            if opp_cell.owner_parity == player_parity:
                result = self._aggressive_chase(board, me, opp, player_parity, rows, cols)
                if result:
                    return result
            # Chase on NEUTRAL cell when very close (mover wins on neutral)
            elif opp_cell.owner_parity == 0 and dist_to_opp <= 2:
                result = self._chase_neutral(board, me, opp, player_parity, rows, cols)
                if result:
                    return result

        # === SIMULATION LAYER: evaluate candidate directions ===
        target = self._find_target_hill(board, me, player_parity, opp_r, opp_c)

        # Get candidate directions with scores
        candidates = self._find_move_candidates(board, me, player_parity, rows, cols,
                                                opp_r, opp_c, danger, near_opp, target,
                                                effective_safe_dist=effective_safe_dist,
                                                powerup_dir=powerup_dir)

        # Determine simulation budget based on time remaining
        tl = time_left()
        if tl > 10:
            sim_turns = 5
            max_candidates = 4
        else:
            sim_turns = 0
            max_candidates = 0

        # Use simulation if we have time and multiple candidates
        if sim_turns > 0 and len(candidates) > 1:
            move_dir = self._simulate_candidates(board, me, player_parity,
                                                 candidates, opp,
                                                 sim_turns, max_candidates)
        elif candidates:
            # No simulation budget — use storm's top pick directly
            move_dir = candidates[0][1]
        else:
            move_dir = None

        return self._build_actions(board, me, player_parity, rows, cols,
                                   opp_r, opp_c, danger, near_opp,
                                   move_dir, target, effective_safe_dist,
                                   chase_danger)

    def _bfs_powerup(self, board, me, parity, danger, near_opp, max_dist=4):
        """BFS to find nearest powerup within max_dist steps."""
        visited = {(me.loc.r, me.loc.c)}
        queue = deque()
        for d in Direction.cardinals():
            nl = me.loc + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            if near_opp and cell.owner_parity != parity:
                continue
            if (nl.r, nl.c) in visited:
                continue
            visited.add((nl.r, nl.c))
            if cell.powerup:
                return d
            queue.append((nl, d, 1))
        while queue:
            pos, first_dir, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for d in Direction.cardinals():
                nl = pos + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in visited:
                    continue
                visited.add((nl.r, nl.c))
                cell = board.cells[nl.r][nl.c]
                if cell.powerup:
                    return first_dir
                queue.append((nl, first_dir, dist + 1))
        return None

    def _bfs_scheduled_powerup(self, board, me, parity, danger, near_opp, max_rounds=2, max_dist=6):
        """Bias movement toward nearby powerups that are about to spawn."""
        schedule = getattr(board, "powerup_schedule", None)
        if not schedule:
            return None

        current_round = getattr(board, "current_round", 0)
        event_pointer = getattr(board, "event_pointer", 0)
        best_dir = None
        best_key = None

        for idx in range(event_pointer, len(schedule)):
            future = schedule[idx]
            rounds_until = future.round_num - current_round
            if rounds_until < 1:
                continue
            if rounds_until > max_rounds:
                break

            path = self._shortest_path_dirs(
                board,
                me.loc.r,
                me.loc.c,
                future.location.r,
                future.location.c,
                max_dist,
            )
            if not path:
                continue
            if len(path) > rounds_until + 2:
                continue

            first_dir = path[0]
            step = me.loc + first_dir
            if board.oob(step):
                continue
            cell = board.cells[step.r][step.c]
            if near_opp and cell.owner_parity != parity:
                continue
            if (step.r, step.c) in danger and cell.owner_parity == -parity:
                continue

            key = (rounds_until, len(path))
            if best_key is None or key < best_key:
                best_key = key
                best_dir = first_dir

        return best_dir

    def _check_powerup(self, board, me, parity, danger, near_opp):
        """Step on adjacent powerup if available."""
        for d in Direction.cardinals():
            nl = me.loc + d
            if board.oob(nl):
                continue
            cell = board.cells[nl.r][nl.c]
            if cell.is_wall:
                continue
            if not cell.powerup:
                continue
            if near_opp and cell.owner_parity != parity:
                continue
            actions = []
            stamina = me.stamina
            if self._can_paint(cell, parity) and stamina >= 30:
                actions.append(Action.Paint(nl))
                stamina -= 15
            actions.append(Action.Move(d))
            return actions
        return None

    def _fortify(self, board, me, parity, rows, cols, opp_r, opp_c):
        actions = []
        stamina = me.stamina

        targets = []
        for d in Direction.cardinals():
            t = me.loc + d
            if board.oob(t):
                continue
            cell = board.cells[t.r][t.c]
            if self._can_paint(cell, parity):
                prio = self._paint_priority(t, cell, parity)
                targets.append((prio, t))

        targets.sort(key=lambda x: -x[0])

        for _, t in targets:
            if stamina < 20:
                break
            actions.append(Action.Paint(t))
            stamina -= 15

        move_dir = self._best_local_move(board, me, parity, rows, cols, opp_r, opp_c)
        if move_dir:
            actions.append(Action.Move(move_dir))
        elif not actions:
            m = self._any_safe_move(board, me, parity, set(), opp_r, opp_c)
            if m:
                actions.append(Action.Move(m))

        return actions

    def _build_actions(self, board, me, parity, rows, cols,
                       opp_r, opp_c, danger, near_opp,
                       move_dir, target, effective_safe_dist, chase_danger):
        actions = []
        stamina = me.stamina
        painted = set()

        if not move_dir:
            for t in self._paintable(board, me.loc, parity, target):
                if stamina < 15:
                    break
                actions.append(Action.Paint(t))
                stamina -= 15
            m = self._any_safe_move(board, me, parity, danger, opp_r, opp_c)
            if m:
                actions.append(Action.Move(m))
            return actions

        step = me.loc + move_dir

        # COLLISION SAFETY (use cached BFS path distance)
        dist_to_opp = self._path_dist_to_opp
        if not board.oob(step):
            step_cell = board.cells[step.r][step.c]

            # SMARTER COLLISION SAFETY: only avoid ENEMY cells near opponent
            # Neutral cells are SAFE (we win collision as the mover)
            if dist_to_opp <= effective_safe_dist and step_cell.owner_parity == -parity:
                # Exception: ERASE on enemy hill cell if we have plenty of stamina
                if (step.r, step.c) in self.hill_set and stamina >= 80:
                    pass  # allow erase below
                else:
                    m = self._any_safe_move(board, me, parity, danger, opp_r, opp_c)
                    if m:
                        actions.append(Action.Move(m))
                    return actions

            # ERASE: if stepping onto enemy-painted hill cell
            if (step.r, step.c) in self.hill_set and step_cell.owner_parity == -parity and stamina >= 80:
                # Paint adjacent targets first
                for t in self._paintable(board, me.loc, parity, target):
                    if stamina < 80:
                        break
                    actions.append(Action.Paint(t))
                    painted.add((t.r, t.c))
                    stamina -= 15
                actions.append(Action.Move(move_dir, move_type=MoveType.ERASE))
                stamina -= 50  # erase cost
                # RETREAT after hill ERASE if opponent is close
                if dist_to_opp <= effective_safe_dist and stamina >= 10:
                    for d in Direction.cardinals():
                        rl = step + d
                        if not board.oob(rl) and rl.r == me.loc.r and rl.c == me.loc.c:
                            if board.cells[rl.r][rl.c].owner_parity == parity:
                                actions.append(Action.Move(d))
                                stamina -= 10
                            break
                return actions

            # LATE-GAME ERASE: enemy border cells for territory expansion
            erase_turn = 30  # ERASE even earlier with strong hill control
            if (step_cell.owner_parity == -parity
                    and not target and self.turn > erase_turn and stamina >= 70):
                for t in self._paintable(board, me.loc, parity, set()):
                    if stamina < 100:
                        break
                    actions.append(Action.Paint(t))
                    painted.add((t.r, t.c))
                    stamina -= 15
                actions.append(Action.Move(move_dir, move_type=MoveType.ERASE))
                stamina -= 50
                return actions

            # SKIP painting destination before moving (paint-after strategy)

        # Paint other adjacent cells (hill priority) — but NOT the destination
        other_targets = self._paintable(board, me.loc, parity, target)
        for t in other_targets:
            if (t.r, t.c) in painted:
                continue
            if stamina < 30:
                break
            actions.append(Action.Paint(t))
            painted.add((t.r, t.c))
            stamina -= 15

        # Move
        actions.append(Action.Move(move_dir))

        # Paint from new position
        if not board.oob(step):
            post_targets = self._paintable(board, step, parity, target)
            for t in post_targets:
                if (t.r, t.c) in painted:
                    continue
                if stamina < 15:
                    break
                actions.append(Action.Paint(t))
                painted.add((t.r, t.c))
                stamina -= 15

        # Multi-move #2 toward target
        if stamina >= 20 and not near_opp and not board.oob(step):
            next_dir = self._find_move(board, me, parity, rows, cols,
                                       opp_r, opp_c, danger, near_opp,
                                       target, from_loc=step,
                                       effective_safe_dist=effective_safe_dist)
            if next_dir:
                next_step = step + next_dir
                if (not board.oob(next_step)
                        and not board.cells[next_step.r][next_step.c].is_wall):
                    nc = board.cells[next_step.r][next_step.c]
                    if (next_step.r, next_step.c) in danger and nc.owner_parity == -parity:
                        return actions  # skip dangerous

                    if self._can_paint(nc, parity) and stamina >= 40:
                        actions.append(Action.Paint(next_step))
                        painted.add((next_step.r, next_step.c))
                        stamina -= 15

                    actions.append(Action.Move(next_dir))
                    stamina -= 10

                    if not board.oob(next_step):
                        p3 = self._paintable(board, next_step, parity, target)
                        for t in p3:
                            if (t.r, t.c) in painted:
                                continue
                            if stamina < 15:
                                break
                            actions.append(Action.Paint(t))
                            painted.add((t.r, t.c))
                            stamina -= 15

                    # Multi-move #3 (enabled on ALL map sizes)
                    if stamina >= 25 and not near_opp:
                        dir3 = self._find_move(board, me, parity, rows, cols,
                                               opp_r, opp_c, danger, near_opp,
                                               target, from_loc=next_step,
                                               effective_safe_dist=effective_safe_dist)
                        if dir3:
                            step3 = next_step + dir3
                            if (not board.oob(step3)
                                    and not board.cells[step3.r][step3.c].is_wall):
                                c3 = board.cells[step3.r][step3.c]
                                if (step3.r, step3.c) in danger and c3.owner_parity == -parity:
                                    return actions
                                if self._can_paint(c3, parity) and stamina >= 40:
                                    actions.append(Action.Paint(step3))
                                    painted.add((step3.r, step3.c))
                                    stamina -= 15
                                actions.append(Action.Move(dir3))
                                stamina -= 20

                                # Multi-move #4 (large maps only)
                                if stamina >= 60 and not near_opp and self.map_tier == 'large':
                                    dir4 = self._find_move(board, me, parity, rows, cols,
                                                           opp_r, opp_c, danger, near_opp,
                                                           target, from_loc=step3,
                                                           effective_safe_dist=effective_safe_dist)
                                    if dir4:
                                        step4 = step3 + dir4
                                        if (not board.oob(step4)
                                                and not board.cells[step4.r][step4.c].is_wall):
                                            c4 = board.cells[step4.r][step4.c]
                                            if (step4.r, step4.c) in danger and c4.owner_parity == -parity:
                                                return actions
                                            if self._can_paint(c4, parity) and stamina >= 55:
                                                actions.append(Action.Paint(step4))
                                                painted.add((step4.r, step4.c))
                                                stamina -= 15
                                            actions.append(Action.Move(dir4))
                                            stamina -= 30

                                            # Paint from move #4 position
                                            if not board.oob(step4):
                                                p4 = self._paintable(board, step4, parity, target)
                                                for t in p4:
                                                    if (t.r, t.c) in painted:
                                                        continue
                                                    if stamina < 15:
                                                        break
                                                    actions.append(Action.Paint(t))
                                                    painted.add((t.r, t.c))
                                                    stamina -= 15

        return actions

    def _paintable(self, board, loc, parity, target):
        hill = []
        normal = []
        for d in Direction.cardinals():
            t = loc + d
            if board.oob(t):
                continue
            cell = board.cells[t.r][t.c]
            if not self._can_paint(cell, parity):
                continue
            is_target = (t.r, t.c) in target
            is_hill = (t.r, t.c) in self.hill_set
            if is_target and cell.owner_parity != parity:
                hill.append(t)
            elif cell.owner_parity == 0:
                normal.append(t)
            # NO reinforcement painting - don't waste stamina on owned cells
        # Deterministic ordering: hills first (sorted by distance to center for consistency)
        return hill + normal

    def _paint_priority(self, loc, cell, parity):
        is_hill = (loc.r, loc.c) in self.hill_set
        if is_hill:
            if cell.owner_parity == 0:
                return 10
            if cell.owner_parity == parity:
                return 8
        if cell.owner_parity == 0:
            return 5
        return 1

    def _find_move(self, board, me, parity, rows, cols,
                   opp_r, opp_c, danger, near_opp, target, from_loc=None,
                   effective_safe_dist=None, powerup_dir=None):
        if effective_safe_dist is None:
            effective_safe_dist = self.safe_dist
        start = from_loc if from_loc else me.loc

        if from_loc is None:
            dist_to_opp = self._path_dist_to_opp
        else:
            dist_to_opp = abs(start.r - opp_r) + abs(start.c - opp_c)

        # If we have a target hill, use direct BFS to find it first
        if target:
            direct = self._bfs_to_target(board, start, parity, rows, cols,
                                         opp_r, opp_c, danger, near_opp, target,
                                         dist_to_opp, effective_safe_dist, me)
            if direct:
                return direct

        # SCORED BFS: for each valid initial direction, count unpainted cells within depth 5
        dirs = list(Direction.cardinals())
        best_dir = None
        best_score = -1

        for d in dirs:
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            # Only avoid stepping on opponent's location if it's THEIR cell
            if nl.r == opp_r and nl.c == opp_c and cell.owner_parity == -parity:
                continue
            # Near opp: only avoid ENEMY cells in danger zone (neutral is safe for us)
            if near_opp and (nl.r, nl.c) in danger and cell.owner_parity == -parity:
                continue
            if dist_to_opp <= effective_safe_dist and cell.owner_parity == -parity:
                continue

            # BFS from this neighbor up to bfs_depth, distance-weighted scoring
            score = 0
            visited = {(start.r, start.c), (nl.r, nl.c)}
            frontier = [(nl.r, nl.c)]
            # Steep weights: heavily favor immediate cells
            steep_weights = [2.0, 1.5, 1.0, 0.5, 0.3]
            weight = steep_weights[0]
            if cell.owner_parity == 0:
                score += weight
                # Frontier bonus: neutral cell next to our territory
                for d3 in Direction.cardinals():
                    adj = nl + d3
                    if not board.oob(adj) and board.cells[adj.r][adj.c].owner_parity == parity:
                        score += 0.4
                        break
            elif cell.owner_parity == -parity:
                score += weight * (0.9 if self.territory_behind else 0.7)
            if (nl.r, nl.c) in self.hill_set and cell.owner_parity != parity:
                score += 15

            # Anti-backtrack
            if from_loc is None and (nl.r, nl.c) in set(self.my_history[-4:]):
                score -= 0.5

            for depth in range(self.bfs_depth - 1):  # adaptive depth
                weight = steep_weights[min(depth + 1, len(steep_weights) - 1)]
                next_frontier = []
                for fr, fc in frontier:
                    for d2 in Direction.cardinals():
                        nl2 = Location(fr, fc) + d2
                        if board.oob(nl2) or board.cells[nl2.r][nl2.c].is_wall:
                            continue
                        if (nl2.r, nl2.c) in visited:
                            continue
                        visited.add((nl2.r, nl2.c))
                        c2 = board.cells[nl2.r][nl2.c]
                        if c2.owner_parity == 0:
                            # Bonus for unpainted cells adjacent to our territory
                            adj_bonus = 0
                            for d3 in Direction.cardinals():
                                adj = Location(nl2.r, nl2.c) + d3
                                if not board.oob(adj) and board.cells[adj.r][adj.c].owner_parity == parity:
                                    adj_bonus += 0.25
                            score += weight + adj_bonus
                        elif c2.owner_parity == -parity:
                            score += weight * (0.9 if self.territory_behind else 0.7)
                        if (nl2.r, nl2.c) in self.hill_set and c2.owner_parity != parity:
                            score += 15
                        next_frontier.append((nl2.r, nl2.c))
                frontier = next_frontier

            # Add small tiebreaker favoring moves away from opponent
            if opp_r >= 0:
                nl_dist = abs(nl.r - opp_r) + abs(nl.c - opp_c)
                score += nl_dist * 0.10

            # Powerup direction bias
            if powerup_dir and from_loc is None and d == powerup_dir:
                score += 0.5

            if score > best_score:
                best_score = score
                best_dir = d

        if best_dir and best_score > 0:
            return best_dir

        # Fallback: any unpainted via regular BFS
        fallback = self._bfs_unpainted(board, start, parity, rows, cols,
                                       opp_r, opp_c, danger, near_opp,
                                       effective_safe_dist)
        if fallback:
            return fallback

        # All painted: seek enemy border
        expand = self._find_enemy_border(board, start, parity, rows, cols, danger, near_opp,
                                         opp_r, opp_c, effective_safe_dist)
        if expand:
            return expand

        # Last resort: chase on owned territory
        if opp_r >= 0:
            chase = self._bfs_toward(board, start, Location(opp_r, opp_c), rows, cols)
            if chase:
                s = start + chase
                if not board.oob(s) and board.cells[s.r][s.c].owner_parity == parity:
                    return chase
        return None

    def _bfs_to_target(self, board, start, parity, rows, cols,
                       opp_r, opp_c, danger, near_opp, target,
                       dist_to_opp, effective_safe_dist, me):
        """BFS specifically for reaching target hill cells."""
        visited = {(start.r, start.c)}
        queue = deque()
        dirs = list(Direction.cardinals())
        # no shuffle
        for d in dirs:
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            if nl.r == opp_r and nl.c == opp_c and cell.owner_parity == -parity:
                continue
            if near_opp and (nl.r, nl.c) in danger and cell.owner_parity == -parity:
                if not (target and (nl.r, nl.c) in target and (nl.r, nl.c) in self.hill_set and me.stamina >= 80):
                    continue
            if dist_to_opp <= effective_safe_dist and cell.owner_parity == -parity:
                if not (target and (nl.r, nl.c) in target and (nl.r, nl.c) in self.hill_set and me.stamina >= 80):
                    continue
            if (nl.r, nl.c) in visited:
                continue
            visited.add((nl.r, nl.c))
            queue.append((nl, d))
        while queue:
            pos, first_dir = queue.popleft()
            cell = board.cells[pos.r][pos.c]
            if (pos.r, pos.c) in target and cell.owner_parity != parity:
                return first_dir
            for d in Direction.cardinals():
                nl = pos + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in visited:
                    continue
                visited.add((nl.r, nl.c))
                queue.append((nl, first_dir))
        return None

    def _bfs_unpainted(self, board, start, parity, rows, cols,
                       opp_r, opp_c, danger, near_opp, effective_safe_dist=None):
        if effective_safe_dist is None:
            effective_safe_dist = self.safe_dist
        dist_to_opp = abs(start.r - opp_r) + abs(start.c - opp_c)
        visited = {(start.r, start.c)}
        queue = deque()
        dirs = list(Direction.cardinals())
        # no shuffle
        for d in dirs:
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            if nl.r == opp_r and nl.c == opp_c and cell.owner_parity == -parity:
                continue
            if near_opp and (nl.r, nl.c) in danger and cell.owner_parity == -parity:
                continue
            if dist_to_opp <= effective_safe_dist and cell.owner_parity == -parity:
                continue
            if (nl.r, nl.c) in visited:
                continue
            visited.add((nl.r, nl.c))
            queue.append((nl, d))
        while queue:
            pos, first_dir = queue.popleft()
            cell = board.cells[pos.r][pos.c]
            if cell.owner_parity == 0 and not cell.is_wall:
                return first_dir
            for d in Direction.cardinals():
                nl = pos + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in visited:
                    continue
                visited.add((nl.r, nl.c))
                queue.append((nl, first_dir))
        return None

    def _find_enemy_border(self, board, start, parity, rows, cols, danger, near_opp,
                           opp_r=-100, opp_c=-100, effective_safe_dist=None):
        """BFS to find enemy cells adjacent to our territory - step on them to weaken."""
        if effective_safe_dist is None:
            effective_safe_dist = self.safe_dist
        dist_to_opp = abs(start.r - opp_r) + abs(start.c - opp_c)
        visited = {(start.r, start.c)}
        queue = deque()
        dirs = list(Direction.cardinals())
        # no shuffle
        for d in dirs:
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            if near_opp and (nl.r, nl.c) in danger and board.cells[nl.r][nl.c].owner_parity == -parity:
                continue
            if dist_to_opp <= effective_safe_dist and board.cells[nl.r][nl.c].owner_parity == -parity:
                continue
            if (nl.r, nl.c) in visited:
                continue
            visited.add((nl.r, nl.c))
            queue.append((nl, d))
        while queue:
            pos, first_dir = queue.popleft()
            cell = board.cells[pos.r][pos.c]
            # Found enemy cell - walk toward it to weaken by stepping on it
            if cell.owner_parity == -parity and not cell.is_wall:
                return first_dir
            for d in Direction.cardinals():
                nl = pos + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in visited:
                    continue
                visited.add((nl.r, nl.c))
                queue.append((nl, first_dir))
        return None

    def _aggressive_chase(self, board, me, opp, parity, rows, cols):
        """Multi-move chase when opponent is on our territory. We win on our cells."""
        chase_dir = self._bfs_toward(board, me.loc, opp.loc, rows, cols)
        if not chase_dir:
            return None
        step = me.loc + chase_dir
        if board.oob(step) or board.cells[step.r][step.c].is_wall:
            return None
        step_cell = board.cells[step.r][step.c]
        if step_cell.owner_parity != parity:
            return None
        actions = []
        stamina = me.stamina
        if self._can_paint(step_cell, parity) and stamina >= 30:
            actions.append(Action.Paint(step))
            stamina -= 15
        actions.append(Action.Move(chase_dir))

        # Multi-move #2: keep chasing on our territory
        if stamina >= 25:
            chase_dir2 = self._bfs_toward(board, step, opp.loc, rows, cols)
            if chase_dir2:
                step2 = step + chase_dir2
                if not board.oob(step2) and not board.cells[step2.r][step2.c].is_wall:
                    step2_cell = board.cells[step2.r][step2.c]
                    if step2_cell.owner_parity == parity:
                        actions.append(Action.Move(chase_dir2))
                        stamina -= 10
                        # Multi-move #3
                        if stamina >= 30:
                            chase_dir3 = self._bfs_toward(board, step2, opp.loc, rows, cols)
                            if chase_dir3:
                                step3 = step2 + chase_dir3
                                if not board.oob(step3) and not board.cells[step3.r][step3.c].is_wall:
                                    step3_cell = board.cells[step3.r][step3.c]
                                    if step3_cell.owner_parity == parity:
                                        actions.append(Action.Move(chase_dir3))
                                        stamina -= 20
        return actions

    def _chase_neutral(self, board, me, opp, parity, rows, cols):
        """Chase opponent on neutral cell. Mover wins collision on neutral."""
        chase_dir = self._bfs_toward(board, me.loc, opp.loc, rows, cols)
        if not chase_dir:
            return None
        step = me.loc + chase_dir
        if board.oob(step) or board.cells[step.r][step.c].is_wall:
            return None
        step_cell = board.cells[step.r][step.c]
        # Safe to step on our cells or neutral cells
        if step_cell.owner_parity == -parity:
            return None
        actions = []
        stamina = me.stamina
        if self._can_paint(step_cell, parity) and stamina >= 30:
            actions.append(Action.Paint(step))
            stamina -= 15
        actions.append(Action.Move(chase_dir))
        return actions

    def _bfs_toward(self, board, start, target, rows, cols):
        visited = {(start.r, start.c)}
        queue = deque()
        for d in Direction.cardinals():
            nl = start + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            if (nl.r, nl.c) in visited:
                continue
            visited.add((nl.r, nl.c))
            if nl.r == target.r and nl.c == target.c:
                return d
            queue.append((nl, d))
        while queue:
            pos, first_dir = queue.popleft()
            for d in Direction.cardinals():
                nl = pos + d
                if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                    continue
                if (nl.r, nl.c) in visited:
                    continue
                visited.add((nl.r, nl.c))
                if nl.r == target.r and nl.c == target.c:
                    return first_dir
                queue.append((nl, first_dir))
        return None

    def _can_paint(self, cell, parity):
        if cell.is_wall or cell.beacon_parity != 0:
            return False
        if cell.owner_parity != 0 and cell.owner_parity != parity:
            return False
        if cell.owner_parity == 0:
            return True
        try:
            return abs(cell.paint_value) < GameConstants.MAX_PAINT_VALUE
        except Exception:
            return abs(cell.paint_value) < 4

    def _best_local_move(self, board, me, parity, rows, cols, opp_r, opp_c):
        best_dir = None
        best_score = -999
        dirs = list(Direction.cardinals())
        # no shuffle
        for d in dirs:
            nl = me.loc + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            if nl.r == opp_r and nl.c == opp_c:
                cell = board.cells[nl.r][nl.c]
                if cell.owner_parity != parity:
                    continue
            score = 0
            for d2 in Direction.cardinals():
                t = nl + d2
                if board.oob(t):
                    continue
                c = board.cells[t.r][t.c]
                if c.is_wall:
                    continue
                if c.owner_parity == 0:
                    score += 3
                elif c.owner_parity == parity and abs(c.paint_value) < GameConstants.MAX_PAINT_VALUE:
                    score += 1
                if (t.r, t.c) in self.hill_set:
                    score += 2
            if score > best_score:
                best_score = score
                best_dir = d
        return best_dir

    def _any_safe_move(self, board, me, parity, danger, opp_r=-100, opp_c=-100):
        dirs = list(Direction.cardinals())
        # no shuffle
        best = None
        best_score = -999
        for d in dirs:
            nl = me.loc + d
            if board.oob(nl) or board.cells[nl.r][nl.c].is_wall:
                continue
            cell = board.cells[nl.r][nl.c]
            score = 0
            if cell.owner_parity == parity:
                score += 10
            elif cell.owner_parity == 0:
                score += 5  # Neutral is safe (we win collision as mover)
            if (nl.r, nl.c) in danger and cell.owner_parity == -parity:
                score -= 50  # Only penalize ENEMY cells in danger
            # Prefer fleeing AWAY from opponent
            if opp_r >= 0:
                new_dist = abs(nl.r - opp_r) + abs(nl.c - opp_c)
                score += new_dist
            if score > best_score:
                best_score = score
                best = d
        return best

    def _count_local(self, board, loc, parity, rows, cols):
        count = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = loc.r + dr, loc.c + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if board.cells[r][c].owner_parity == parity:
                        count += 1
        return count

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
