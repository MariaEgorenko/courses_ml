def search_substr(substr: str, st: str) -> str:
    
    return "Есть контакт!" if substr in st else "Мимо!"

str = "Machine learning"
print(search_substr("learning", str))
print(search_substr("python", str))