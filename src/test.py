import functools
import multiprocessing
from multiprocessing import Process, Value, Array
def func(arr, i):
    # print  arr[i][1]+3
    a= arr[i][0] + 3
    arr[i][0] = arr[i][0] + 3
    print  (a)
    print(arr[i][0])

if __name__ == '__main__':
    manager = multiprocessing.Manager()  # Create a manager to handle shared object(s).
    xyz = manager.list([manager.list([1,2]),manager.list([3,2])])  # Create a proxy for the shared list object.

    p = multiprocessing.Pool(processes=4)  # Create a pool of worker processes.

    # Create a single arg function with the first positional argument (arr) supplied.
    # (This is necessary because Pool.map() only works with functions of one argument.)
    mono_arg_func = functools.partial(func, xyz)
    # print len(xyz)
    p.map(mono_arg_func, range(len(xyz)))  # Run func in parallel until finished

    print (xyz)
        