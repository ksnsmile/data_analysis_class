# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Library, Numpy
# =============================================================================


### numpy

x=[10, 20, 30]
y=x*x
y=x**2


import numpy as np

x = np.array([10, 20, 30])
print(type(x))

y= x**2 
print(type(y))

y=x*2


a=np.array([10, 20, 30]) 
print(a)
print(type(a))

a.shape 
a.ndim 
a.dtype 
a.itemsize 
a.size 


a=np.array([10, 20, 30])
a.dtype 

a[2]=40.1
a.dtype  
print(a) 

a.dtype.name
a = a.astype('float64') 
print(a)

a[2]=30.1
print(a)

c=np.array([10.1, 20.1, 30.1], 'int32')
print(c)

x=np.array([7, 9, 11])
y=x/2

x.dtype
y.dtype


a=np.array([10, 20, 30])
a=np.array([10, 
            20, 
            30])

a.ndim 
b=np.array([10, 'abc', 30]) 


a=np.array([[10, 20, 30],[40, 50, 60]])
a=np.array([[10, 20, 30],
            [40, 50, 60]])

a.ndim 
a.shape 
a.size 

a=np.array([[10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]])

a.ndim 
a.shape 
a.size 

a=np.array([[[10],[20]],[[30],[40]]])
a=np.array(
  [
   [[10],[20]],
   [[30],[40]]
   ]  
   )

a.ndim 
a.shape 
a.size 

a=np.array(
  [
   [[10,50],[20,60]],
   [[30,70],[40,80]]
   ]  
   )
a.shape 

a1=np.array([10, 
             20, 
             30])

b1=np.array([[10, 20, 30],
             [40, 50, 60]])

c1=np.array([
             [
              [10, 20, 30],
              [40, 50, 60]
             ],
             [
              [11, 21, 31],
              [41, 51, 61]
             ]
            ])

a=np.array([10, 20, 30])
a=np.insert(a, 0, 5)

a= np.delete(a,0)
a

a=np.array([1, 2, 3]) 
a=np.arange(3) 
a=np.zeros((2,3)) 
a=np.ones((2,3)) 
a=np.linspace(0,5,6)
a=np.logspace(0,5,11)


a=np.array([10, 20, 30]) 
b=np.array([1, 2, 3]) 

c=a+b 
c=a-b
c=b**2
c=2*a

a = a+1
a += 1
a *= 2

result = b < 20
print(result)
result = a < 20

a=np.array([10, 20, 30]) 
b=np.array([1, 2, 3]) 

np.add(a,b) 
np.subtract(a,b) 
np.multiply(a,b) 
np.divide(a,b) 
c=np.divmod(a,b) 
print(c)
np.exp(b)
print(np.exp(b))
np.sqrt(b)
print(np.sqrt(b))

a=np.array([10, 20, 30])
 
np.mean(a) 
a.mean() 

np.average(a)
np.average(a, weights=[1,1,1]) 
np.average(a, weights=[1,1,0]) 
np.average(a, weights=[0,1,1])
np.average(a, weights=[1,0,1])

np.median(a) 
np.cumsum(a)
np.std(a) 
np.var(a) 


x=np.array([10, 20, 30]) 
y=x.sum() 

x=np.array([10, 20, 30, 25, 15]) 
x.min() 
x.argmin() 

x.max() 
x.argmax() 

x_min, x_min_idx = x.min(), x.argmin()
x_max, x_max_idx = x.max(), x.argmax()

x.ptp() 
x.sort() 

x=np.array([10, 20, 30, 25, 15]) 
y=np.sort(x) 
idx=np.argsort(x)

a=np.array([10, 20, 30]) 
b=np.array([-5, 25]) 

np.searchsorted(a,b)

d=np.arange(1,7,1) 
d.shape


d.reshape(2,3) 
e=d.reshape(2,3)
f=np.linspace(1,10,10) 
g=np.linspace(1,10,10).reshape(2,5)


a=np.array([[1,2],[3,4]])
np.repeat(a,2) 
np.repeat(a,2,axis=0) 
np.repeat(a,2,axis=1) 


a=np.array([[1],[2],[3]])
b=np.array([[4],[5],[6]])
b.shape

np.concatenate((a,b), axis=0) 
np.concatenate((a,b), axis=1) 
np.concatenate((a,b), axis=2)

a=np.array([[[1,10],[2,20],[3,30]]])
b=np.array([[[4,40],[5,50],[6,60]]])
b.shape

np.concatenate((a,b), axis=0) 
np.concatenate((a,b), axis=1) 
np.concatenate((a,b), axis=2)

a=np.array([10, 20, 30]) 
b=np.array([40, 50, 60]) 

np.vstack((a,b))   
np.hstack((a,b))   

A=np.array([[10, 20, 30],
           [40, 50, 60]])

np.hsplit(A,3)
np.vsplit(A,2)

B=np.vsplit(A,2)
print(type(B))

A=np.array([[10, 20, 30],
           [40, 50, 60]])

A.transpose()

A.ravel()
A.reshape(-1)
A.flatten()

A=np.array([[1,2,3]])
A.shape 
B=A.squeeze()
B.shape

A=np.array([10, 20, 30]) 
B=A

B[0]=99
print(A)
print(B)

B is A

A=np.array([10,20,30,40])
B=A.view()

B is A 

B[0]=99 
print(A)
print(B)


A=np.array([10,20,30,40])
B=A.copy()

B is A

B[0]=99
print(A)
print(B)


a=np.array([0,1,2])
a.all()

a.any()

a.nonzero()

a=np.array([100,0,10,20])
np.where(a>0)
np.where(a==0)

M1=np.array([100.,
             101.,
             102.,
             103.,
             104.])
M1.shape
M1.ndim


M1[0]
print(M1[0])
M1[1]
M1[2]
M1[-1]
M1[-5]

M1[0:3] 
M1[0:3:1]
M1[0:3:2] 
M1[:] 
M1[::1] 
M1[::2] 
M1[::-1]


M2=np.array([[100,101,102],[200,201,202]])

M2.shape
M2.ndim


M2[0][0] 
M2[0][2] 

M2[0,2] 
M2[1,1] 

M2[:] 
M2[:,:] 
M2[0,:] 
M2[:,0]
M2[:,[0,2]]

x=np.array([10,20,30])
idx=np.where(x==30) 
x[idx]

idx=(x==30) 
idx 
x[idx]






























































