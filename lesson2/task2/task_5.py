import sys

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

def find_queen_moves(coor: str) -> list:

    x, y = coor2lst(coor)

    directions = [
        (0, 1),   # вверх
        (0, -1),  # вниз
        (1, 0),   # вправо
        (-1, 0),  # влево
        (1, 1),   # вправо-вверх (по диагонали)
        (1, -1),  # вправо-вниз
        (-1, 1),  # влево-вверх
        (-1, -1), # влево-вниз
    ]

    result = []
    for x1, y1 in directions:
        x2, y2 = x + x1, y + y1
        while 0 <= x2 < 8 and 0 <= y2 < 8:
            result.append(coor2str([x2, y2]))
            x2 += x1
            y2 += y1

    return result

my_fig = input('Введите координаты вашей фигуры: ')
enemy_fig = input('Введите координаты вражекой фигуры: ')

queen_moves = find_queen_moves(enemy_fig)
knigh_moves = find_knigh_moves(enemy_fig)

menace = False

if my_fig in queen_moves:
    menace = True
    print("Угроза, если вражеская фигура ферзь!")
if my_fig in knigh_moves:
    menace = True
    print("Угроза, если вражеска фигура конь!")

if not menace:
    print("Угрозы не обнаружено!")