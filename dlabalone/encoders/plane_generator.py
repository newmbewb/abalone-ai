from dlabalone.abltypes import Direction


def _can_push(board, stone, length, direction):
    # stone: the player's first stone
    # length: the length of the stone row
    player = board.grid.get(stone)
    next_point = stone + direction
    all_vuln_points = []
    first_next_point = next_point
    opp_player_count = 0
    valid = None
    if not board.is_on_grid(next_point):
        return False, first_next_point, opp_player_count == length - 1, all_vuln_points
    for _ in range(length - 1):
        if not board.is_on_grid(next_point):
            return True, first_next_point, opp_player_count == length - 1, all_vuln_points
        elif board.grid.get(next_point, None) == player:
            return False, first_next_point, opp_player_count == length - 1, all_vuln_points
        elif board.grid.get(next_point, None) == player.other:
            opp_player_count += 1
        all_vuln_points.append(next_point)
        next_point = next_point + direction
    if board.grid.get(next_point, None) is None:
        return True, first_next_point, opp_player_count == length - 1, all_vuln_points
    else:
        return False, first_next_point, opp_player_count == length - 1, all_vuln_points


def _is_danger_point(board, point, length):
    if length == 2:
        return board.distance_from_center(point) >= board.size - 1
    elif length == 3:
        return board.distance_from_center(point) >= board.size - 2
    else:
        assert False, 'Wrong length'


def calc_opp_layers(board, next_player):
    _next_player = next_player
    opp_player = _next_player.other
    vuln_points = {}
    vuln_stones = {2: set(), 3: set()}
    danger_stones = set()
    danger_points = set()
    opp_sumito_stones = set()
    for length in [2, 3]:
        for direction in Direction:
            vuln_points[(length, direction.value)] = []
    for head_point in type(board).valid_grids:
        player = board.grid.get(head_point, None)
        if player != opp_player:
            continue
        for string_direction in [Direction.EAST.value, Direction.SOUTHEAST.value, Direction.SOUTHWEST.value]:
            for length in range(2, 3 + 1):
                # Make stone list
                stones = [head_point]
                next_point = head_point
                for i in range(1, length):
                    next_point = next_point + string_direction
                    stones.append(next_point)

                # Check stones' player
                valid = True
                for stone in stones[1:]:
                    if opp_player != board.grid.get(stone, None):
                        valid = False
                        break
                if not valid:
                    continue

                # Check whether it can push
                directions = [string_direction, (string_direction * -1)]
                first_stones = [stones[-1], stones[0]]
                # for direction in directions:
                for first_stone, direction in zip(first_stones, directions):
                    # print(stones)
                    # print(direction)
                    valid, first_next_stone, minimum_sumito_stone, all_vuln_points_cur = \
                        _can_push(board, first_stone, length, direction)
                    if valid:
                        vuln_points[(length, direction)].append(first_next_stone)
                        if minimum_sumito_stone:
                            for stone in stones:
                                opp_sumito_stones.add(stone)
                        add_vuln_stone = board.grid.get(first_next_stone, None) == _next_player

                        # Calculate danger stones
                        # All point should be at edge, and all points are next player's stone
                        add_danger_stone = True
                        for point in all_vuln_points_cur:
                            if not _is_danger_point(board, point, length) or \
                                    not (board.grid.get(point, None) == _next_player):
                                add_danger_stone = False
                                break
                        if add_danger_stone:
                            for point in all_vuln_points_cur:
                                danger_stones.add(point)

                        # Calculate other danger_points and vuln_stones
                        for point in all_vuln_points_cur:
                            if _is_danger_point(board, point, length) and \
                                    _is_danger_point(board, first_next_stone, length):
                                danger_points.add(point)
                            if add_vuln_stone and board.grid.get(point, None) == _next_player:
                                vuln_stones[length].add(point)

    return opp_sumito_stones, danger_stones, danger_points, vuln_stones, vuln_points


def count_opp_sumito(board, next_player):
    opp_player = next_player.other
    sumito_count = 0
    # for head_point in type(board).valid_grids:
    for head_point, player in board.grid.items():
        if player != opp_player:
            continue
        for string_direction in [Direction.EAST.value, Direction.SOUTHEAST.value, Direction.SOUTHWEST.value]:
            for length in range(2, 3 + 1):
                # Make stone list
                stones = [head_point]
                next_point = head_point
                for i in range(1, length):
                    next_point = next_point + string_direction
                    stones.append(next_point)

                # Check stones' player
                valid = True
                for stone in stones[1:]:
                    if opp_player != board.grid.get(stone, None):
                        valid = False
                        break
                if not valid:
                    continue

                # Check whether it can push
                directions = [string_direction, (string_direction * -1)]
                first_stones = [stones[-1], stones[0]]
                # for direction in directions:
                for first_stone, direction in zip(first_stones, directions):
                    # print(stones)
                    # print(direction)
                    valid, first_next_stone, minimum_sumito_stone, all_vuln_points_cur = \
                        _can_push(board, first_stone, length, direction)
                    if valid:
                        if minimum_sumito_stone:
                            sumito_count += 1

    return sumito_count


def count_unique_push_moves(moves):
    moves_three = []
    moves_two = {}
    for move in moves:
        if len(move.stones) == 3:
            moves_three.append(move)
        if len(move.stones) == 2:
            moves_two[frozenset(move.stones)] = move.direction

    redundant_count = 0
    for move in moves_three:
        stones1 = frozenset(move.stones[1:])
        stones2 = frozenset(move.stones[:-1])
        if stones1 in moves_two and moves_two[stones1] == move.direction:
            redundant_count += 1
        if stones2 in moves_two and moves_two[stones2] == move.direction:
            redundant_count += 1

    return len(moves) - redundant_count
