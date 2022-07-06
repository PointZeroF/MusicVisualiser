import numpy as np
import sys
import data
import psutil
import os

def transformData(frame, shader, outsize):
    import wgpu.backends.rs
    from wgpu.utils import compute_with_buffers

    #repeat for all channels
    for j in range(0, 3):
        #format input correctly
        inputdata = np.ascontiguousarray(frame[:, :, j], dtype=np.uint32)

        #assign input

        inputs = {0: inputdata, 2: mapping}

        #compute GPU shader
        ########### MEMORY LEAK 2MB/s ########### - not my fault - use threads and kill for gc?
        outputdata = compute_with_buffers(inputs, {1: inputdata.nbytes}, shader, n=(outsize, outsize, 1))

        #save output
        temp = np.reshape(np.frombuffer(outputdata[1], dtype=np.uint32), (outsize, outsize))
        frame[:, :, j] = temp

    del inputdata
    del outputdata
    del temp

#get arguments
argv = sys.argv
lengthinframes = int(argv[1])
tempfolderpath = str(argv[2])
memmappath = str(argv[3])
i = int(argv[4])
spiral = int(argv[5]) == 1 

mismatch = str(argv[6]) == "True"
outsize = 90 if spiral else 64

#load correct shader
shader = data.compute_shader_spiral if spiral else data.compute_shader_hilbert
#load correct map
mapping = data.maps.spiral if spiral else data.maps.hilbert

#load inputs from file
if mismatch:
    framemap = np.memmap(tempfolderpath.format("_FRAMES.memmap"), dtype=np.uint8, mode="r", shape=(lengthinframes, 90, 90, 3))
else:
    framemap = np.memmap(tempfolderpath.format("_FRAMES.memmap"), dtype=np.uint8, mode="r", shape=(lengthinframes, outsize, outsize, 3))

transformedmap = np.memmap(memmappath, dtype=np.uint8, mode="r+", shape=(lengthinframes, outsize, outsize, 3))

#monitor memory used
process = psutil.Process(os.getpid())
rss = process.memory_info().rss / (1024 ** 3)

#quit if we finish or use to much memory
while rss < 0.5 and i < lengthinframes:
    #copy data from framesmap
    if mismatch:
        frame1 = np.copy(framemap[i, :, :, :])
        frame2 = frame1.reshape(8100, 3)
        frame3 = frame2[0:4096, :]
        frame = frame3.reshape(outsize, outsize, 3)
    else:
        frame = np.copy(framemap[i, :, :, :])

    #transform, write
    transformData(frame, shader, outsize)

    #save to file
    transformedmap[i, :, :, :] = frame[:, :, :]
    transformedmap.flush()

    i += 1
    rss = process.memory_info().rss / (1024 ** 3)

transformedmap._mmap.close()

#if we arent done, report location and exit 1
if i < lengthinframes:
    print(i)
    sys.exit(1)
else:
    #otherwise exit zero
    print("DONE")
    sys.exit(0)