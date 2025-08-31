import collections

def count_it(sequence: str) -> dict:

    if not sequence.isdigit():
        return {}
    
    lst = [int(num) for num in sequence]
    cnt = collections.Counter(lst)
    return dict(cnt.most_common(3))

st = input('Введите строку из чисел:\n')

print(count_it(st))
