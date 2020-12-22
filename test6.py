import time
import os
base = r'E:\Dane\input\sample_images'
start = time.time()
for i in os.listdir(base):
    print(i)
end = time.time()
print(end - start)