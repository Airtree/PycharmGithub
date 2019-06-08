import numpy as np
import sortedcontainers as sc
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

import mysql.connector

import collections

a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])

print(a[:2:, :])
print(a[2::, :])

b = np.array([[1, 2, 3]])
print("Vector shape description: " + str(b.shape))

aMat = np.mat('1 2; 3 4')
bMat = np.mat('4 3; 2 1')
print(np.sum(a, axis=None))  # axis = None is default accumulating all elements in the matrix
print(np.sum(a, axis=0, keepdims=True))  # sum row,
print(np.sum(a, axis=1, keepdims=True))  # sum col
sum0 = np.sum(a, axis=0, keepdims=True)
sum1 = np.sum(a, axis=1, keepdims=True)
print(np.dot(sum1, b))
matMul = aMat * bMat  # standard matrix multiplication
print('Matrix multiplication' + str(matMul))
print('Square of array' + str(np.sum(np.square(a))))
print('Test sum： ' + str(np.sum(a)))

print("hello everyone!")

a.reshape(2, -1)
print("a: " + str(a))

print(a.reshape(2, -1))
var = np.eye(5)[b.reshape(-1)].shape
print(var)
print("test : " + str(np.eye(5)[(3,)]))

'''
    Labmda
'''
X = Input(shape=(4, 5))
print(X)
# x = Lambda(lambda x: aMat[:, 1])(aMat)

'''
    List and List Comprehension
'''

listExample1 = [1, 2, 3, 4]
listExample2 = [2, 3, 4, 5]
listComp1 = [x ** 2 for x in listExample1]
listComp2 = [x for x in listExample1 for y in listExample2 if x == y]
listComp3 = [(x, y) for x in listExample1 for y in listExample2 if x != y]
listComp4 = [[x ** 2, y ** 3] for x in listExample1 for y in listExample2]

print("ListComp1: " + str(listComp1))
print("ListComp2: " + str(listComp2))
print("ListComp3: " + str(listComp3))
print("ListComp4: " + str(listComp4))

'''
    Dictionary 
'''
Dic = {"Brand": "BMW", "Model": "520", "Year": 2018}
print(Dic)
for x in Dic:
    print(Dic[x])  # print values of all keys
    print(x)  # print keys

for key, value in Dic.items():
    print(key, value)  # print pair of key and value

# Add element in Dic
Dic["Color"] = "White"  # Add new element for the Dictinoary

for key, value in Dic.items():
    print(key, value)  # print pair of key and value

# Remove element in Dic
del Dic["Color"]
Dic.pop("Year")
Dic.popitem()  # remove the last element in the dictionary
print(Dic)

# Using dict constructor to construct a dictionary
DicConstructor = dict(Brand="BMW", Model="520", Year=2018)
print(DicConstructor)

'''
        Class
'''


class Person:
    def __init__(this, _name, _age):
        this.name = _name
        this.age = _age

    def pickName(this):
        print("My name is " + this.name)


p = Person("Nick", "30")
p.pickName()

'''
        Iterable vs Iterator
'''


class MyNumber:
    def __iter__(this):
        this.a = 0
        return this

    def __next__(this):
        if this.a < 10:
            this.a += 1
        else:
            raise StopIteration
        return this.a


num = MyNumber()
numItr = iter(num)
print("Iterator class: " + str(next(numItr)))
print("Iterator class: " + str(next(numItr)))
for x in num:
    print("Iterator class: " + str(x))
'''
range()
'''
for x in range(2, 6):
    print(x)

'''
        Recursion
'''


def tri_recursion(k):
    if (k > 0):
        result = k + tri_recursion(k - 1)
        print(result)
    else:
        return 0
    return result


print("tri_recursion: ")
tri_recursion(6)

'''
            Lambda function
'''


def Lamb_func(n):
    return lambda a: a * n


lb = Lamb_func(10)
print(lb(2))  # this time run lambda function output 20

'''
            map   map(function_object, iterable1, iterable2,...)
'''


def multiply2(x):
    return 2 * x


mapList1 = map(multiply2, [1, 2, 3, 4])
mapList2 = map(lambda x: 2 * x, [1, 2, 3, 4])  # lambda form

print("mapList: " + str(list(mapList2)))

dictList = [{"name": "Python", "Points": 80}, {"name": "C++", "Points": 100}, {"name": "C++", "Points": 99}]

mapDic1 = map(lambda x: x["name"], dictList)
mapDic2 = map(lambda x: x["name"] == "Python", dictList)
mapDic3 = map(lambda x, y: x + y, [1, 2, 3, 4], [5, 6, 7, 8])
print("mapDic1: " + str(list(mapDic1)))
print("mapDic2: " + str(list(mapDic2)))
print("mapDic3: " + str(list(mapDic3)))

'''
            Filter filter(function_object, iterable), function_object must return bool.
'''
filterDic1 = filter(lambda x: x["name"] == "C++", dictList)
print("filterDic1: " + str(list(filterDic1)))

'''
            for loop copy issues
'''
words = ['cat', 'window', 'defenestrate']
for w in words[:]:  # if no slice operator [:], words will not be copied, infinite loop will happen
    if len(w) > 6:
        words.insert(0, w)
print(words)

'''
            Sorted Set    
'''

ss = sc.SortedSet([5, 3, 9, 2, 3])
print("Sorted Set: " + str(ss))
ss.add(6)
print(ss)

'''
    Ordered Dictionary
'''
od = sc.SortedDict({"name": "BMW", "year": "1981"})
print(od)
od.update({"color": "red"})
print(od)

'''
    sorted() function in terms of key
'''


class LocalMap:
    def __init__(this, _xValue):
        this.x = _xValue;


def comparison(m):
    return m.x


myList = [LocalMap(10), LocalMap(4), LocalMap(30)]

LocalMapSorted = sorted(myList, key=comparison)
var = [m.x for m in LocalMapSorted]
print("LocalMapSorted: " + str(var))

import numpy as np

arr = np.array([1, 3, 2, 4, 5])
print(arr.argsort()[-3:][::-1])  # argsort returns index,

'''
    Decorator
'''


def our_decorator(func):
    def func_wrapper(*args, **kwargs):
        print("Before decorator called: " + func.__name__)
        func(*args, **kwargs)
        print("After decorator called: " + func.__name__)
        return func

    return func_wrapper


@our_decorator
def foo(x):
    print("foo is been called! " + str(x))


# def foo(x):
#     print("foo is been called! " + str(x))
#
#
# foo = our_decorator(foo)


def foo1():
    print("foo1 is been called! No parameter!")


foo("BMW")
foo1()


def greeting(expr):
    def greeting_decorator(func):
        def function_wrapper(x):
            print(expr + ", " + func.__name__ + " returns:")
            func(x)

        return function_wrapper

    return greeting_decorator


@greeting("καλημερα")
def boo(x):
    print(42)


boo("Hi")

import logging


def use_logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == "warn":
                logging.warning("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


@use_logging(level="warn")
def fooName(name='fooName'):
    print("i am %s" % name)


fooName()

'''
    Pass by reference 
'''


def try_to_change_list_contents(the_list):
    print('got', the_list)
    the_list.append('four')
    print('changed to', the_list)


outer_list = ['one', 'two', 'three']

print('before, outer_list =', outer_list)
try_to_change_list_contents(outer_list)
print('after, outer_list =', outer_list)


def try_to_change_list_reference(the_list):
    print('got', the_list)
    the_list = ['and', 'we', 'can', 'not', 'lie']
    print('set to', the_list)


outer_list = ['we', 'like', 'proper', 'English']

print('before, outer_list =', outer_list)
try_to_change_list_reference(outer_list)
print('after, outer_list =', outer_list)


def checkRef(integer):
    integer.append(0)


integer = [1, 2, 3]
checkRef(integer)
print(integer)


def treeRecur(s, countLeft, countRight, existing, res, n):
    if (countLeft == n and countRight == n):
        res.append(existing)
        return

    tmp = existing + s
    if s == '(':
        countLeft += 1
    if s == ')':
        countRight += 1
    if countLeft == n:
        treeRecur(')', countLeft, countRight, tmp, res, n)
    elif countLeft > countRight:
        treeRecur('(', countLeft, countRight, tmp, res, n)
        treeRecur(')', countLeft, countRight, tmp, res, n)
    elif countLeft == countRight:
        treeRecur('(', countLeft, countRight, tmp, res, n)


def generateParenthesis(n):
    """
    :type n: int
    :rtype: List[str]
    """
    countLeft = 0
    countRight = 0
    res = []
    existing = ""
    treeRecur('(', countLeft, countRight, existing, res, n)
    return res


l = generateParenthesis(3)
print(l)

'''
    one liner
'''

with open("Resignation.txt") as fh:
    var = sum(1 for line in fh for character in line if character.isupper())

# print(var)


'''
    Randomize
'''
from random import shuffle

x = ['Keep', 'The', 'Blue', 'Flag', 'Flying', 'High']
shuffle(x)
print(x)

print([0 for i in range(10)])
'''
    MySQL
'''


# mydb = mysql.connector.connect(
#   host='localhost:3306',
#   user='root',
#   passwd='(yiqu1144huidOu-)',
# )


def get_data_attrs_names(data):
    """
     get the categorical inputs and numerical inputs and output column names.
     We use the dtype of the data to decide between numerical and categorical attribut

     arguements:
     data -- pandas dataframe.

     return:
     a dict with the following keys and values
     y: list of of the target attribute
     X_cat: list of categorical inputs i.e dtype == object
     X_num: list of numerical inputs i.e dtype != object
     X: list of the combied {X_num, X_cat} inputs
    """

    all_attribs = list(data.columns.values)
    target_attrib = ['y']

    # seperate out output column
    y = data[target_attrib]
    print("\n\nsample of 'target i.e output attributes'")
    print(y.head())

    data_cat = data.select_dtypes(include=['object']).copy()
    data_cat = data_cat.drop(target_attrib[0], axis=1)
    cat_attribs = list(data_cat.columns.values)
    print("\n\n'sample of categorical attribute and output attributes'")
    print(data_cat.head())

    # sep out continous aka scale columns
    data_scale = data.select_dtypes(include=[np.number]).copy()
    num_attribs = list(data_scale.columns.values)
    print("\n\n'sample of continous (scale) attributes'")
    print(data_scale.head())

    # col_X contains all the predictors
    X_attribs = list(all_attribs)  # copy all cols names
    X_attribs.remove(target_attrib[0])

    res = {
        'y': target_attrib,
        'X_cat': cat_attribs,
        'X_num': num_attribs,
        'X': X_attribs
    }
