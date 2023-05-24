# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:12:47 2023

@author: Chunhui TU

---------------------------
FILE DESCRIPTION:
---------------------------
This file contains 2 tools: 
    1. monitor the resource usage when the program is running
    2. calculate the resource consumption when the program is running
"""

import time
import psutil
import threading


def monitor_Resource_Usage():
    # Start monitoring
    cpu_percent_list = []
    memory_usage_list = []
    
    # Event to signal thread termination
    stop_event = threading.Event()
    
    def monitor():
        while not stop_event.is_set():
            # Get CPU usage
            cpu_percent = psutil.cpu_percent()
            cpu_percent_list.append(cpu_percent)
            
            # Get memory usage
            memory_usage = psutil.virtual_memory().used
            memory_usage_list.append(memory_usage)
            
            ## Print current resource usage
            #print(f"CPU Usage: {cpu_percent}% | Memory Usage: {memory_usage} bytes")
            
            # Adjust the sleep interval based on your desired monitoring frequency
            time.sleep(1)  # Sleep for 1 second
    
    # Start monitoring in a separate thread
    thread = threading.Thread(target=monitor)
    thread.start()
    
    # Return the monitoring thread and stop event
    return thread, stop_event, cpu_percent_list, memory_usage_list


def calculate_Resource_Usage(cpu_percent_list, memory_usage_list, time_total):
    
    cpuBenchmark = cpu_percent_list[-1]
    memBenchmark = memory_usage_list[-1]
    
    # delete the benchmark element
    cpu_percent_list.pop()
    memory_usage_list.pop()
    
    # subtract the usage when the program is not running
    cpu_usage_list = [x - cpuBenchmark for x in cpu_percent_list] 
    # calculate CPU and MEM average usage
    cpu_avg = sum(cpu_usage_list)/time_total
    # subtract the usage when the program is not running
    mem_usage_list = [x - memBenchmark for x in memory_usage_list] 
    mem_avg = sum(mem_usage_list)/time_total
    # convert mem_avg from bytes to MB
    mem_avg = mem_avg/(1024 ** 2)
    
    return round(cpu_avg, 3), round(mem_avg, 3)
    

