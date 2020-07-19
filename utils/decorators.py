import time

def timer(function):

	def wrapper(*args, **kwargs):
		start_time = time.time()
		res = function(*args, **kwargs)
		print("Time : {} seconds".format(time.time()-start_time))
		return res

	return wrapper

