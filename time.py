import time

while True:
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(current_time, end='\r')  # The '\r' character moves the cursor to the beginning of the line
    time.sleep(1)  # Update every second
