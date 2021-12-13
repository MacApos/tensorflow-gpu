from matplotlib import pyplot as plt

filename = 'naca4412'

dat = open(filename + '.dat', 'r')
geo = open(filename + '.geo', 'w')
ellipse = open('Elipse.geo', 'r')

# lines = [line for line in dat.readlines() if line[0].isnumeric()]
# line = '1.0000     0.0013\n'


lines = []

for line in dat.readlines():
    if line[0].isnumeric():
        for i in line.split(' '):
            print(i)




#
# multiplier = 2
# angle = -10
#
#
# def join(text, first_line, last_line):
#     result = "".join(text[first_line:last_line])
#     return result
#
#
# ellipse = [line for line in ellipse]
# first_chunk = join(ellipse, 0, 24)
# line_loop = [line for line in ellipse if line.startswith("Line Loop")][0]
# second_chunk = join(ellipse, [ellipse.index(line) for line in ellipse if line.startswith("Plane Surface")][0]-1,
#                     len(ellipse))
#
# geo.write(first_chunk)
#
#
# lines = dat.readlines()
#
# x = []
# y = []
#
# line_counter = 0
#
# for line in lines:
#     if not line[:1].isalpha():
#         data = [string for string in line.split(' ') if string]
#         line_counter += 1
#         x.append(float(data[0])*multiplier)
#         y.append(float(data[1])*multiplier)
#
# j = 5
# k = j
#
# points = []
# lines = []
# spline = []
# line_loop2 = []
#
# for i in range(line_counter):
#     spline.append(str(j))
#     line_loop2.append(str(j))
#     point = "Point(%i) = {%.4f, %.4f, 0};\n" % (j, x[i], y[i])
#     geo.write(point)
#     if i < line_counter-1:
#         line = "Line(%i) = {%i,%i};\n" % (j, j, j+1)
#     else:
#         line = "Line(%i) = {%i,%i};\n" % (j, j, k)
#     lines.append(line)
#     j += 1
#
# for line in lines:
#     geo.write(line)
#
# spline.append(str(k))
# spline = "//Spline(5) = {%s};\n" % ", ".join(spline)
# geo.write(spline)
#
# geo.write(line_loop)
#
# line_loop2_str = "Line Loop(2) = {%s};\n" % ", ".join(line_loop2)
# geo.write(line_loop2_str)
# transfinite_line = "Transfinite Curve{%s}=35 Using Progression 1.0;\n" % ", ".join(line_loop2)
# geo.write(transfinite_line)
#
# rotate = '//Rotate {{0, 0, 1}, {0, 0, 0}, Pi/%i} {Curve{5}; }\n' % (180/angle)
# geo.write(rotate)
#
# geo.write(second_chunk)