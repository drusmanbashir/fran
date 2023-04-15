from datetime import datetime

def append_time(input_str, now=True):
    now = datetime.now()
    dt_string = now.strftime("_%d%m%y_%H%M")
    return input_str+dt_string

