def infinite_sequence():
    num = 0
    while True:
        num += 1
        # print(num)
        yield num
        if num >= 10:
            num = 1


gen = infinite_sequence()
print(next(gen))
