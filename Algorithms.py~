import math
import numpy as np
import hello as he


def permuteRecur(sTmp, flags, res, s):
    '''

    :param sTmp:  string
    :param flags: a list of boolean
    :param res:   results will be stored in List<string>
    :param s:     original string
    :return:      void
    '''

    if len(sTmp) == len(s):
        res.append(sTmp)
        return

    for i in range(len(s)):
        if flags[i]:
            continue

        flags[i] = True
        sTmp += s[i]
        permuteRecur(sTmp, flags, res, s)
        sTmp = sTmp[:-1]
        flags[i] = False


def permuteString(s):
    '''

    :param s:  the string s must have no duplicate characters!!!
    :return: List of string
    '''

    if s == '':
        return []
    res = []
    flags = [False] * len(s)
    permuteRecur("", flags, res, s)
    return res


st = "asbt"
res = permuteString(st)

print(res)
print("All permutations of the string " + st + " have been validated as Right(True)/Wrong(False): "
      + str(math.factorial(len(st)) == len(res)))


def jeweryStone(J, S):
    ref = np.zeros((1, 256), dtype=int)
    count = 0
    for ch in J:
        ref[0, ord(ch)] = 1
    print(ref)

    for s in S:
        if ref[0, ord(s)] == 1:
            count += 1
    return count


J = 'aA'

S = 'aAabcd'

print(jeweryStone(J, S))

with open('txtSamples.txt') as f:
    lines = [line.rstrip('\n') for line in f]
    for l in lines:
        output = [s for s in l.split('"') if s.strip() != '']
        print(output)


def splitting(l):
    pass
