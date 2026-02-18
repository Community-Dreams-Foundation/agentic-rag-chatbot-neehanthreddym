import time

# Timer decorator function
def timer(base_function):
    def enhanced_function(*args, **kwargs):
        start_time = time.time()
        result = base_function(*args, **kwargs)
        end_time = time.time()
        print(f"Task time: {end_time - start_time} seconds")
        return result
    return enhanced_function