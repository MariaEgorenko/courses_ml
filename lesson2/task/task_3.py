lst = [1, 2, 3, 4, 5]

def change(lst):

    # поменять местами первый и последний элемент списка
    lst[0], lst[-1] = lst[-1], lst[0]
    return lst

print(change(lst))
