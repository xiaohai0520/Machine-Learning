import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# get a random seed
np.random.seed(2)

# the loc is 3.0 and scale is 1 and 100 numbers at total
pageSpeeds = np.random.normal(3.0,1.0,100)

# get another 100numbers to divide the each number in pagespeed
purchaseAmount = np.random.normal(50.0,30.0,100)/pageSpeeds

# get five subplot in the figure
# 1-2: all numbers
# 3-4: train and test of predict
fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax3 = fig1.add_subplot(223)
ax4 = fig1.add_subplot(224)

# draw the scatter picture with page as x and purchase as y
ax1.scatter(pageSpeeds,purchaseAmount)


# get the 80% trainx and 20% test  of x
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

# get the 80% trainx and 20% test  of y
trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

# draw the scatter picture with page as x and purchase as y of train and test
ax2.scatter(trainX, trainY, c='r')
ax2.scatter(testX, testY, c='g')
# plt.show()

# get x y np arrays
x = np.array(trainX)
y = np.array(trainY)

#polyfit(x,y,degree) to calculate the coefficients
#poly1d() get the polynomial
p4 = np.poly1d(np.polyfit(x, y, 6))
print(p4)

# creat a xp array as the input
xp = np.linspace(0, 7, 100)
# set the contributes of the pitcute
ax3.set_xlim(0,6)
ax3.set_ylim(0,200)
# draw the input point
ax3.scatter(x, y)
# draw the predict curve
ax3.plot(xp, p4(xp), c='r')

# test is as same sa train
testx = np.array(testX)
testy = np.array(testY)
ax4.set_xlim(0,6)
ax4.set_ylim(0,200)
ax4.scatter(testx,testy)
ax4.plot(xp, p4(xp), c='r')


# use r2_score to make sure whether the predict
# curve is good enough
r2_train = r2_score(y,p4(x))
print("train:", r2_train)

r2_test = r2_score(testy, p4(testx))
print("test:",r2_test)

plt.show()



