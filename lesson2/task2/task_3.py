import collections

def def_th_frequent_characters(str: str) -> list:
    
    counter = collections.Counter(str)
    del counter[' ']
    return counter.most_common(3)

str = input("Введите строку для подсчета символов:\n")
print("Топ 3 часто встречаемых символов:", def_th_frequent_characters(str))