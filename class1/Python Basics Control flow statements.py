# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Basics, Control flow statements
# =============================================================================


### if statements

money=False
if money:
    print("택시를 타고 가라")
else:
    print("걸어 가라")
    
    
money=5000
if money >= 3000:
    print("택시를 타고 가라")
    print("택시비는 내가 줄게")
else:
    print("걸어 가라")
   
      
money=5000
card=True
if money >= 3000 or card:
    print("택시를 타고 가라")
else:
    print("걸어 가라")
    

if 1 not in [1,2,3]:
    print("택시를 타고 가라")
else:
    print("걸어 가라")
    
    
poket  = ['paper', 'cellphone']
if 'money' in poket:
    print("택시를 타고 가라")
else:
    print("걸어 가라")


if 'money' in poket:
   pass
else:
    print("걸어 가라")
  
    
pocket  = ['paper', 'cellphone']
card = 1
if 'money' in pocket:
   print("택시를 타고 가라")
else:
    if card:
        print("택시를 타고 가라")
    else:
        print("걸어 가라")    


if 'money' in pocket:
    print("택시를 타고 가라")
elif card:
    print("역시 택시를 타고 가라")
else:
    print("걸어 가라")
    
     
poket  = ['money', 'paper', 'cellphone']
card = True
others = True
if 'money' in poket:
    print("현금 택시를 타고 가라")
elif card:
    print("카드 택시를 타고 가라")
elif others:
    print("가지 마라") 
else:
    print("걸어 가라")
    

### while statements
    
treeHit = 0
while treeHit < 10:
    treeHit = treeHit +1 
    print("나무를 %d번 찍었습니다." %treeHit)
    if treeHit == 10:
        print("나무 넘어갑니다.")
 
      
coffee = 10
money = 3000
while money:
    print("돈을 받았으니 커피를 줍니다")
    coffee = coffee-1
    print("남은 커피의 양은 %d개입니다." % coffee)
    if coffee==0:
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
        break
   
    
a=0
while a<10:
    a=a+1
    if a%2 == 0: continue
    print(a)


### for statements
    
test_list = ['one', 'two', 'three']
for i in test_list:
    print(i)
    

a=[(1,2), (3,4), (5,6)]
for (first, last) in a:
    print(first+last)
 
    
marks = [90, 25, 67, 45, 80]
number = 0
for mark in marks:
    number = number+1
    if mark >= 60:
        print("%d번 학생은 합격입니다" % number)
    else:
        print("%d번 학생은 불합격입니다" % number)
   

marks = [90, 25, 67, 45, 80]
number = 0
for mark in marks:
    number = number+1
    if mark < 60: continue
    print("%d번 학생 축하합니다. 합격입니다." % number)

     
sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)


for i in range(2,10):
    for j in range(1,10):
        print(i*j, end=" ")
    print('')
  

a=[1,2,3,4]
result=[]
for num in a:
    result.append(num*3)
print(result)

result1=[num*3 for num in a]
print(result1)


a=[1,2,3,4]
result3=[]
for num in a:
    if num % 2 == 0:
        result3.append(num*3)
print(result3)

result4=[num*3 for num in a if num % 2 == 0]
print(result4)