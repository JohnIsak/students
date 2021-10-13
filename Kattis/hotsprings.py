import functools
import math

rand = input()
x = list(map(int, input().split()))
#print(x)
x.sort()
#print(x)
j = len(x)-1
i = 0
max_diff = 999999
ans = list()
n = 0


while i <= j:
    ans.append(x[i])
    i += 1
    n += 1
    if (n >= 2):
        diff = abs(ans[n-1]-ans[n-2])
        if diff > max_diff:
            print("IMPOSSIBLE")
            break
        max_diff = diff
        if i > j:
            break
    ans.append(x[j])
    j -= 1
    n += 1
    diff = abs(ans[n-1] - ans[n-2])
    if diff > max_diff:
        print("IMPOSSIBLE")
        break
    max_diff = diff

ans.reverse()
print(*ans)







