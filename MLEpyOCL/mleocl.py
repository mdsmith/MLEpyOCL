import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import os

SITES = 1024
CHARACTERS = 64
NODES = 200


def mleOCL():
    # Platform test
    if len(cl.get_platforms()) > 1:
        for found_platform in cl.get_platforms():
            if found_platform.name == 'NVIDIA CUDA':
                my_platform = found_platform
                print "Selected platform:", my_platform.name
    else: my_platform = cl.get_platforms()[0]
    
    for device in my_platform.get_devices():
      dev_type = cl.device_type.to_string(device.type)
      if dev_type == 'GPU':
            dev = device
            print "Selected device: ", dev_type
    
    # context
    ctx = cl.Context([dev])
    #ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    kernel = defKernel()

    node_cache = np.array([1.0/CHARACTERS] * CHARACTERS*SITES, dtype=np.float32)
    parent_cache = np.array([1.0] * CHARACTERS*SITES, dtype=np.float32)
    scalings_cache = np.array([0] * CHARACTERS*SITES, dtype=np.int32)
    model = np.array([1.0/CHARACTERS] * CHARACTERS*CHARACTERS, dtype=np.float32)

    node_array = cla.to_device(queue, node_cache)
    parent_array = cla.to_device(queue, parent_cache)
    scalings_array = cla.to_device(queue, scalings_cache)
    model_array = cla.to_device(queue, model)

    uflowthresh = 0.00000001
    scalar = 100.0

    prg = cl.Program(ctx, kernel).build()
    
    for i in range(NODES):
        event = prg.FirstLoop(queue, (CHARACTERS*SITES,), (CHARACTERS,), 
                node_array.data, 
                model_array.data, 
                parent_array.data,
                np.int32(NODES),
                np.int32(SITES),
                np.int32(CHARACTERS),
                scalings_array.data, 
                np.float32(uflowthresh),
                np.float32(scalar))
        event.wait()

    parent_cache = parent_array.get()
    print(parent_cache)

def defKernel():
    return """
typedef float fpoint;

__kernel void FirstLoop(__global const fpoint* node_cache, 
                        __global const fpoint* model, 
                        __global fpoint* parent_cache, 
                        int nodes, 
                        int sites, 
                        int characters, 
                        __global int* scalings, 
                        fpoint uflowthresh, 
                        fpoint scalar
                        ) {
    // get index into global data array
    int parentCharGlobal = get_global_id(0); // a unique global ID for each parentcharacter
    int parentCharLocal = get_local_id(0); // a local ID unique within the site.

    __local fpoint nodeScratch[64];
    __local fpoint modelScratch[64];
    nodeScratch[parentCharLocal] = node_cache[parentCharGlobal];
    modelScratch[parentCharLocal] = model[parentCharLocal * characters + parentCharLocal];
    barrier(CLK_LOCAL_MEM_FENCE);

    fpoint sum = 0.;
    long myChar;
    for (myChar = 0; myChar < characters; myChar++) {   
        sum += nodeScratch[myChar] * modelScratch[myChar];     
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    while (parent_cache[parentCharGlobal] < uflowthresh) {
        parent_cache[parentCharGlobal] *= scalar;
        scalings[parentCharGlobal] += 1;
    }       
    parent_cache[parentCharGlobal] *= sum;
}
    """

if __name__ =="__main__":
    print os.environ['PYOPENCL_CTX']
    mleOCL()
