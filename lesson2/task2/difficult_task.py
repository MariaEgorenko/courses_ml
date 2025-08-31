import sys
import random

def coor2lst(coor: str) -> list:

    col, row = coor[0], coor[1]
    x = ord(col) - ord('a')  
    y = int(row) - 1

    if not (0 <= x < 8 and 0 <= y < 8):
        print("Некорректная позиция")
        sys.exit(0)

    return [x, y]

def coor2str(coor: list) ->str:
    col = chr(ord('a') + coor[0])
    row = str(coor[1] + 1)
    return col + row

def find_knigh_moves(coor: str) -> list:
    
    x, y = coor2lst(coor)

    moves = [
        (2, 1), (2, -1),
        (-2, 1), (-2, -1),
        (1, 2), (1, -2), 
        (-1, 2), (-1, -2)
    ]

    result = []
    for x1, y1 in moves:
        x2, y2 = x + x1, y + y1
        if 0 <= x2 < 8 and 0 <= y2 < 8:
            result.append(coor2str([x2, y2]))

    return result

my_fig = input("Введите координаты вашей фигуры: ")
enemy_fig = input("Введите координаты вражекой фигуры: ")

knigh_moves = find_knigh_moves(my_fig)
if enemy_fig in knigh_moves:
    print("Угроза для вражеской фигуры!")
else:
    print("Угрозы для вражеской фигуры нет!")

print("Выполняется ход конем...")
my_fig = random.choice(knigh_moves)
print("Ход выполнен на", my_fig)

if my_fig == enemy_fig:
    print("Вражеская фигура была уничтожена!")
else:
    knigh_moves = find_knigh_moves(my_fig)
    if enemy_fig in my_fig:
        print("Угроза для вражеской фигуры!")
    else:
        print("Угрозы для вражеской фигуры нет!")
