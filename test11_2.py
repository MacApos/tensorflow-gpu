src = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Literatura.txt'
dst = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Literatura2.txt'

with open(dst, 'w') as file:
    file.write('Literatura\n')

with open(src, 'r') as file:
    lines = file.readlines()
    for line in sorted(lines):
        print(line, end='')

for idx, line in enumerate(lines):
    with open(dst, 'a') as file:
        file.write('[{}] {}'.format(idx+1, line))

