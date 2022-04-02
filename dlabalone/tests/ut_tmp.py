from dlabalone.ablboard import Move

move1 = Move([(0, 0), (1, 1), ], (1, 1))
move2 = Move([(1, 1), (0, 0), ], (0, 1))

print(move1 == move2)
print(move1)
print(move2)

s1 = str(move1)
s2 = str(move2)
print(Move.str_to_move(s1))
print(Move.str_to_move(s2))