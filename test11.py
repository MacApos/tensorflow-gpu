src1 = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Bibliografia.txt'
src2 = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Artykuły.txt'
src3 = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\linki.txt'
dst = r'F:\Nowy folder\7\Praca inżynierska\Dokumenty\Literatura.txt'

with open(src1, 'r') as file:
    lines1 = file.readlines()
    for line in lines1:
        print(line, end='')

with open(src2, 'r') as file:
    lines2 = file.readlines()
    for line in lines2:
        print(line, end='')

with open(src3, 'r') as file:
    lines3 = file.readlines()
    for line in lines3:
        print(line, end='')

# for line in lines2:
#     print(line)

with open(dst, 'w') as file:
    file.write('Literatura\n')

for idx, line in enumerate(sorted(lines1)):
    with open(dst, 'a') as file:
        file.write(line)

for idx, line in enumerate(sorted(lines2)):
    with open(dst, 'a') as file:
        file.write(line)

for idx, line in enumerate(sorted(lines3)):
    with open(dst, 'a') as file:
        file.write('[{}] {}'.format(idx+7,line))