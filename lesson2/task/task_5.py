lst = ['крот', 'белка', 'выхухоль']

def all_eq(lst: list) -> list:
    
    max_len_str = max(len(s) for s in lst)
    result = []

    for s in lst:
        filling = '_' * (max_len_str - len(s))
        result.append(s + filling)

    return result

print(all_eq(lst))
