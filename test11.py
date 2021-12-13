src1 = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Bibliografia.txt'
dst = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Literatura.txt'

with open(dst, 'w') as file:
    file.write('')

with open(src1, 'r') as file:
    lines = file.readlines()
    for line in sorted(lines):
        print(line, end='')

for idx, line in enumerate(sorted(lines)):
    with open(dst, 'a') as file:
        file.write(line)

