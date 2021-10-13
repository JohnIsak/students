
x = input()
n = x.count('e')
#n = [val == 'e' for val in x].count()
ans = list()
i = 0
added = False
while i < len(x):
    if x[i] == 'e' and not added:
        for j in range(n):
            ans.append('e')
        added = True
    ans.append(x[i])
    i += 1

print("".join(ans))



