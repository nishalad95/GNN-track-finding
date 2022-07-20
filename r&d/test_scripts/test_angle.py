from math import *


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a

def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(atan2(deltaY, deltaX))

angle = getAngleBetweenPoints(5, 2, 1,4)
print(angle)
assert angle >= 0, "angle must be >= 0"
angle = getAngleBetweenPoints(1, 1, 2, 1)
print(angle)
assert angle == 0, "expecting angle to be 0"
angle = getAngleBetweenPoints(2, 1, 1, 1)
print(angle)
assert abs(pi - angle) <= 0.01, "expecting angle to be pi, it is: " + str(angle)
angle = getAngleBetweenPoints(2, 1, 2, 3)
print(angle)
assert abs(angle - pi/2) <= 0.01, "expecting angle to be pi/2, it is: " + str(angle)
angle = getAngleBetweenPoints(2, 1, 2, 0)
print(angle)
assert abs(angle - (pi+pi/2)) <= 0.01, "expecting angle to be pi+pi/2, it is: " + str(angle)
angle = getAngleBetweenPoints(1, 1, 2, 2)
print(angle)
assert abs(angle - (pi/4)) <= 0.01, "expecting angle to be pi/4, it is: " + str(angle)
angle = getAngleBetweenPoints(-1, -1, -2, -2)
print(angle)
assert abs(angle - (pi+pi/4)) <= 0.01, "expecting angle to be pi+pi/4, it is: " + str(angle)
angle = getAngleBetweenPoints(-1, -1, -1, 2)
print(angle)
assert abs(angle - (pi/2)) <= 0.01, "expecting angle to be pi/2, it is: " + str(angle)