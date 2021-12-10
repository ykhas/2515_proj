from project.timer import Timer

ti = Timer(timer_repeat_times= 100000)
# TODO Radian
# The lambda must return something, or else crash
ti.time_average(lambda :  "Hello World!")
print(ti.str_average())
