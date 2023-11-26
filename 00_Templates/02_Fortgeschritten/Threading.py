# https://www.educative.io/answers/what-is-threadingactivecount-in-python
# Program to count active threads
# active_count() method from Threading Module
import threading
import time
# Methods for three threads..
def thread1_Subroutine(i):
    time.sleep(2)
    print("Thread-1: Number of active threads:", threading.active_count())
    print('Thread 1 Value:', i)

def thread2_Subroutine(i):
    print("Thread-2: Number of active threads:", threading.active_count())
    print('Thread 2 Value:', i)

def thread3_Subroutine(i):
    time.sleep(5) # suspends execution for 5 seconds
    print("Thread-3: Number of active threads:", threading.active_count())
    print("Thread 3 Value:", i)

# Creating sample threads
thread1 = threading.Thread(target=thread1_Subroutine, args=(100,), name="Thread1")
thread2 = threading.Thread(target=thread2_Subroutine, args=(200,), name="Thread2")
thread3 = threading.Thread(target=thread3_Subroutine, args=(300,), name="Thread3")
print("START: Current active thread count: ", threading.active_count())
# Calling start() method to initialize execution
thread1.start()
thread2.start()
thread3.start()
thread3.join() # Wait for thread-3 to join.