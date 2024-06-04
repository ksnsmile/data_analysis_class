# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Basics, Data Types
# =============================================================================


### Basics

a=3
print(a)
print(type(a))

a=3
b=4
print(a+b)
print(a*b)
print(a/b)
print(a%b)
print(a//b)
print(a**b)

# food='Python's favorite food is perl'    # Error
food="Python's favorite food is perl"
food='Python\'s favorite food is perl'

head = "python"
tail = " is fun!"
print(head+tail)

a = 'python'
a*2

a = "Life is too short, You need Python"
a[0:4]

a = "20010331Rainy"
date = a[:8]
weather = a[8:]

print(date)
print(weather)

print(a[::])
print(a[::2])
print(a[::-2])

print("I eat %d apples" %3)

number=10
day="three"
print("I ate %d apples. so I was sick for %s days" %(number, day))

a= "I eat {number} apples.".format(number=3)
print(a)

print("%10s" % "hi")
print("%-10sjane." % "hi")

print("%0.4f" % 3.43234234)
print("%10.4f" % 3.43234234)

a = "hobby"
a.count('b')

a ="Python is best choice"
a.find('b')
a.find('k')

a = "Life is too short"
a.index('t')
a.index('k')

a=','
a.join('abcd')
"".join(['a','b','c','d'])
"_".join(['a','b','c','d'])

a = 'hi'
a.upper()
b = 'HI'
b.lower()

a = ' hi '
a.strip()
a.rstrip()
a.lstrip()

a = "Life is too short"
a.replace("Life", "Your leg")

a.split()
a = "a:b:c:d"
a.split(":")


### List

a = "이주연"
b = "홍길동"
c = "이대감"
d = "저대감"
e = "그대감"

print(a)

a=["이주연", "홍길동", "이대감", "저대감", "그대감"]

e=[1,2,['Life', 'is']]
print(e[2][0])

a = [1,2,3]
a[0]
a[0]+a[2]
a[-1]

a = [1,2,3,4,5]
a[0:2]

b = a[:2]
c = a[2:]

print(b)
print(c)

a = [1,2,3]
b = [4,5,6]

a+b
a*3

a = [1,2,3]
a[2]=4
print(a)

print(a[1:2])
a[1:2] = ['a', 'b', 'c']
print(a)

a[1:3] = []
print(a)
del a[1]
print(a)

a = [1,2,3]
a.append(4)
print(a)

a = [1, 4, 3, 2]
a.sort()
print(a)

a = ['a', 'c', 'b']
a.reverse()
print(a)

a = [1,2,3]
a.index(3)
a.index(0)

a.insert(0,4)
print(a)

a = [1,2,3,1,2,3]
a.remove(3)
print(a)
a.remove(3)
print(a)

a = [1,2,3]
a.pop()
print(a)
del a[0]
print(a)

a = [1,2,3,1]
a.count(1)

a = [1,2,3]
a.extend([4,5])
print(a)
b = [6,7]
a.extend(b)
print(a)


### Tuple

t1=(1,2,'a','b')
del t1[0]

t1[0]
t1[3]

t1[1:]

t2 = (3,4)
print(t1+t2)

t2*3
print(t2*3)


### Dictionary

a = {1:'a'}
a[2]='b'
a['name'] = 'pey'
a[3] = [1,2,3]

print(a)

del a[1]
print(a)

grade = {'pey': 10, 'julliet': 99}
grade['pey']
grade['julliet']

a = {1:'a', 1:'b'}
print(a)

a = {'name': 'pey', 'phone': '0119993323', 'birth': '1118'}
a.keys()
b=list(a.keys())
a.values()

for v in a.keys():
    print(v)

a.items()

a.clear()
print(a)

a = {'name': 'pey', 'phone': '0119993323', 'birth': '1118'}
a.get('name')
a.get('Juyeon')
a['Juyeon']
a.get('foo', 'bar')
a.get('Juyeon', 'NA')

'name' in a
'email' in a
'email' not in a

# a = ['a', 'c', 'b']
# 'b' in a
# 'd' in a
# 'd' not in a


### Set

s1 = set([1,2,3])
s2 = {1,2,3}
s3 = set("Hello")
print(s3)

s1 = set([1,2,3,4,5,6])
s2 = set([4,5,6,7,8,9])

print(s1&s2)
s1.intersection(s2)

print(s1|s2)
s1.union(s2)

print(s1-s2)
s1.difference(s2)

print(s2-s1)
s2.difference(s1)

s1 = set([1,2,3])
s1.add(4)
s1

s1 = set([1,2,3])
s1.update([4,5,6])
s1

s1 = set([1,2,3])
s1.remove(2)
s1


### bool

a = [1, 2, 3, 4]
while a:
    b=a.pop()
    print(b)

a = [1, 2, 3, 4]
while a:
    a.pop()
    print(a)


### variable

a = [1,2,3]
b=a
a[1]=4

print(a)
print(b)

print(id(a))
print(id(b))


b=a[:]
a[0]=10

print(a)
print(b)

print(id(a))
print(id(b))

from copy import copy

b=copy(a)
a[1]=0

print(a)
print(b)

print(id(a))
print(id(b))



























