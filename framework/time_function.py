import time
from datetime import timedelta

class TimeUtil:    
    def time_function_execution(function_to_execute):
        def compute_execution_time(*args, **kwargs):
            start_time = time.time()
            result = function_to_execute(*args, **kwargs)
            end_time = time.time()
            computation_time = timedelta(seconds=end_time - start_time)
            print(function_to_execute.__name__ , ' function took: {}'.format(computation_time), flush=True)
            return result
        return compute_execution_time

#@TimeUtil.time_function_execution
#def say_hi():
#    time.sleep(2)
#    print('sleeping...')
#    
#say_hi()   