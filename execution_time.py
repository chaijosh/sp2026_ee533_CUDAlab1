import sys, os

def execute_system_call(command_str, N):
    # Added this function for debugging any TLE, SegFaults, errors etc.
    # as matmult on CPU can fill up stack/heap very quickly which causes aborts
    try:
        exit_code = os.system(command_str)
        if exit_code != 0:
            print(f"Command failed with exit code: {exit_code} for N={N}")
            exit()
    except TypeError as e:
        print(f"An exception related to argument type occurred: {e} for N={N}")
        exit()



N = ['512','1024', '2048', '4096', '8192']
for n in N:
    for i in range (0,5):
        execute_system_call('./matrix_gpu ' + n + ' >> matrix_gpu.csv',n)
        execute_system_call('./matrix_gpu_tiled ' + n + ' >> matrix_gpu_tiled.csv',n)
        execute_system_call('./matrix_gpu_cuBLAS ' + n + ' >> matrix_gpu_cuBLAS.csv',n)

# Putting long GPU test and all CPU tests at the end so I an atleast collect data for smaller N values without error
for i in range (0,5):
    execute_system_call('./matrix_gpu ' + '16384' + ' >> matrix_gpu.csv',n)
    execute_system_call('./matrix_gpu_tiled ' + '16384' + ' >> matrix_gpu_tiled.csv',n)
    execute_system_call('./matrix_gpu_cuBLAS ' + '16384' + ' >> matrix_gpu_cuBLAS.csv',n)

for n in N:
    for i in range (0,5):
        execute_system_call('./matrix_cpu ' + n + ' >> matrix_cpu.csv',n)