/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of the class responsible for executing the N-Body simulation, provide update, and final data set
*/

#import "AAPLSimulation.h"
#import "AAPLKernelTypes.h"
#import "AAPLMathUtilities.h"

/// Utility function providing a random with which to initialize simulation
static vector_float3 generate_random_normalized_vector(float min, float max, float minlength)
{
    vector_float3 rand;

    do
    {
        rand = generate_random_vector(min, max);
    } while(vector_length(rand) > minlength);

    return vector_normalize(rand);
}

@implementation AAPLSimulation
{
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLComputePipelineState> _computePipeline;

    // Metal buffer backed with memory wrapped in an NSData object for updating client (renderer)
    id<MTLBuffer> _updateBuffer;

    // Wrapper for system memory used to transfer to client (renderer)
    NSData *_updateData;

    // Two buffers to hold positions and velocity.  One will hold data for the previous/initial
    // frame while the other will hold data for the current frame, which is generated using data
    // from the previous frame.
    id<MTLBuffer>  _positions[2];
    id<MTLBuffer>  _velocities[2];

    MTLSize _dispatchExecutionSize;
    MTLSize _threadsPerThreadgroup;
    NSUInteger _threadgroupMemoryLength;

    // Indices into the _positions and _velocities array to track which buffer holds data for
    // the previous frame  and which holds the data for the new frame.
    uint8_t _oldBufferIndex;
    uint8_t _newBufferIndex;

    id<MTLBuffer> _simulationParams;

    // Current time of the simulation
    CFAbsoluteTime _simulationTime;

    const AAPLSimulationConfig  * _config;
}

/// Initializer used to create a simulation from the beginning
- (instancetype)initWithComputeDevice:(nonnull id<MTLDevice>)computeDevice
                               config:(nonnull const AAPLSimulationConfig *)config
{
    self = [super init];

    if(self)
    {
        _device = computeDevice;

        _config = config;

        [self createMetalObjectsAndMemory];

        [self initializeData];
    }

    return self;
}

/// Initializer used to continue a simulation already begun on another device
- (nonnull instancetype)initWithComputeDevice:(nonnull id<MTLDevice>)computeDevice
                                       config:(nonnull const AAPLSimulationConfig *)config
                                 positionData:(nonnull NSData *)positionData
                                 velocityData:(nonnull NSData *)velocityData
                            forSimulationTime:(CFAbsoluteTime)simulationTime
{
    self = [super init];

    if(self)
    {
        _device = computeDevice;

        _config = config;

        [self createMetalObjectsAndMemory];

        [self setPositionData:positionData
                 velocityData:velocityData
            forSimulationTime:simulationTime];
    }
    return self;
}

/// Initialize Metal objects and set simulation parameters
- (void)createMetalObjectsAndMemory
{
    // Create compute pipeline for simulation
    {
        NSError *error = nil;

        // Load all the shader files with a .metal file extension in the project
        id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

        id<MTLFunction> nbodySimulation = [defaultLibrary newFunctionWithName:@"NBodySimulation"];

        _computePipeline = [_device newComputePipelineStateWithFunction:nbodySimulation error:&error];
        if (!_computePipeline)
        {
            NSLog(@"Failed to create compute pipeline state, error %@", error);
        }
    }

    // Calculate parameters to efficiently execute the simulation kernel
    {
        _threadsPerThreadgroup = MTLSizeMake(_computePipeline.threadExecutionWidth, 1, 1);
        _dispatchExecutionSize =  MTLSizeMake(_config->numBodies, 1, 1);
        _threadgroupMemoryLength = _computePipeline.threadExecutionWidth * sizeof(vector_float4);
    }

    // Create buffers to hold our simulation data and generate initial data set
    {
        NSUInteger bufferSize = sizeof(vector_float3) *  _config->numBodies;

        // Create 2 buffers for both positions and velocities since we'll need to preserve previous
        // frames data while computing the next frame
        for(int i = 0; i < 2; i++)
        {
            _positions[i] = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeManaged];
            _velocities[i] = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeManaged];

            _positions[i].label = [NSString stringWithFormat:@"Positions %i", i];
            _velocities[i].label = [NSString stringWithFormat:@"Velocities %i", i];
        }
    }

    // Setup buffer of simulation parameters to pass to compute kernel
    {
        _simulationParams = [_device newBufferWithLength:sizeof(AAPLSimParams) options:MTLResourceStorageModeManaged];

        _simulationParams.label = @"Simulation Params";

        AAPLSimParams *params = (AAPLSimParams *)_simulationParams.contents;

        params->timestep = _config->simInterval;
        params->damping = _config->damping;
        params->softeningSqr = _config->softeningSqr;
        params->numBodies = _config->numBodies;

        [_simulationParams didModifyRange:NSMakeRange(0, _simulationParams.length)];
    }

    // Create buffers to transfer data to our client (i.e. the renderer)
    {
        NSUInteger updateDataSize = _config->renderBodies * sizeof(vector_float3);

        // Allocate buffer with page aligned address
        void *updateAddress;
        kern_return_t err = vm_allocate((vm_map_t)mach_task_self(),
                                        (vm_address_t*)&updateAddress,
                                        updateDataSize,
                                        VM_FLAGS_ANYWHERE);

        assert(err == KERN_SUCCESS);

        _updateBuffer = [_device newBufferWithBytesNoCopy:updateAddress
                                                      length:updateDataSize
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];

        _updateBuffer.label = @"Update Buffer";

        // Wrap the memory allocated with vm_allocate with an NSData object which will allow
        // use to rely on ObjC ARC (or even MMR) to manage the memory's lifetime

        // Block to deallocate memory created with vm_allocate when the NSData object is no
        // longer referenced
        void (^deallocProvidedAddress)(void *bytes, NSUInteger length) =
            ^(void *bytes, NSUInteger length)
            {
                vm_deallocate((vm_map_t)mach_task_self(),
                              (vm_address_t)bytes,
                              length);
            };

        // Create a data object to wrap system memory and pass a deallocator to free the
        // memory allocated with vm_allocate when the data object has been released
        _updateData = [[NSData alloc] initWithBytesNoCopy:updateAddress
                                                      length:updateDataSize
                                                 deallocator:deallocProvidedAddress];
    }
}

/// Set the initial positions and velocities of the simulation based upon the simulation's config
- (void)initializeData
{
    const float pscale = _config->clusterScale;
    const float vscale = _config->velocityScale * pscale;
    const float inner  = 2.5f * pscale;
    const float outer  = 4.0f * pscale;
    const float length = outer - inner;

    _oldBufferIndex = 0;
    _newBufferIndex = 1;

    vector_float4 *positions = (vector_float4 *) _positions[_oldBufferIndex].contents;
    vector_float4 *velocities = (vector_float4 *) _velocities[_oldBufferIndex].contents;

    for(int i = 0; i < _config->numBodies; i++)
    {
        vector_float3 nrpos    = generate_random_normalized_vector(-1.0, 1.0, 1.0);
        vector_float3 rpos     = generate_random_vector(0.0, 1.0);
        vector_float3 position = nrpos * (inner + (length * rpos));

        positions[i].xyz = position;
        positions[i].w = 1.0;

        vector_float3 axis = {0.0, 0.0, 1.0};

        float scalar = vector_dot(nrpos, axis);

        if((1.0f - scalar) < 1e-6)
        {
            axis.xy = nrpos.yx;

            axis = vector_normalize(axis);
        }

        vector_float3 velocity = vector_cross(position, axis);

        velocities[i].xyz = velocity * vscale;
    }

    NSRange fullRange;
    fullRange = NSMakeRange(0, _positions[_oldBufferIndex].length);
    [_positions[_oldBufferIndex] didModifyRange:fullRange];
    fullRange = NSMakeRange(0, _velocities[_oldBufferIndex].length);
    [_velocities[_oldBufferIndex] didModifyRange:fullRange];
}

/// Set simulation data for a simulation that was begun elsewhere (i.e. on another device)
- (void)setPositionData:(nonnull NSData *)positionData
           velocityData:(nonnull NSData *)velocityData
      forSimulationTime:(CFAbsoluteTime)simulationTime
{
    _oldBufferIndex = 0;
    _newBufferIndex = 1;

    vector_float4 *positions = (vector_float4 *) _positions[_oldBufferIndex].contents;
    vector_float4 *velocities = (vector_float4 *) _velocities[_oldBufferIndex].contents;

    assert(_positions[_oldBufferIndex].length == positionData.length);
    assert(_velocities[_oldBufferIndex].length == velocityData.length);

    memcpy(positions, positionData.bytes, positionData.length);
    memcpy(velocities, velocityData.bytes, velocityData.length);

    NSRange fullRange;
    fullRange = NSMakeRange(0, _positions[_oldBufferIndex].length);
    [_positions[_oldBufferIndex] didModifyRange:fullRange];
    fullRange = NSMakeRange(0, _velocities[_oldBufferIndex].length);
    [_velocities[_oldBufferIndex] didModifyRange:fullRange];

    _simulationTime = simulationTime;
}

/// Blit a subset of the positions data for this frame and provide them to the client
/// to show a summary of the simulation's progress
- (void)fillUpdateBufferWithPositionBuffer:(nonnull id<MTLBuffer>)buffer
                        usingCommandBuffer:(nonnull id<MTLCommandBuffer>)commandBuffer
{
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    blitEncoder.label = @"Position Update Blit Encoder";

    [blitEncoder pushDebugGroup:@"Position Update Blit Commands"];

    [blitEncoder copyFromBuffer:buffer
                   sourceOffset:0
                       toBuffer:_updateBuffer
              destinationOffset:0
                           size:_updateBuffer.length];

    [blitEncoder popDebugGroup];

    [blitEncoder endEncoding];
}

/// Run a frame of the simulation with the given command buffer (used both when simulation is run
/// synchronously or asynchronously)
- (nonnull id<MTLBuffer>)simulateFrameWithCommandBuffer:(nonnull id<MTLCommandBuffer>)commandBuffer
{
    [commandBuffer pushDebugGroup:@"Simulation"];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    computeEncoder.label = @"Compute Encoder";

    [computeEncoder setComputePipelineState:_computePipeline];

    [computeEncoder setBuffer:_positions[_newBufferIndex] offset:0 atIndex:AAPLComputeBufferIndexNewPosition];
    [computeEncoder setBuffer:_velocities[_newBufferIndex] offset:0 atIndex:AAPLComputeBufferIndexNewVelocity];
    [computeEncoder setBuffer:_positions[_oldBufferIndex] offset:0 atIndex:AAPLComputeBufferIndexOldPosition];
    [computeEncoder setBuffer:_velocities[_oldBufferIndex] offset:0 atIndex:AAPLComputeBufferIndexOldVelocity];
    [computeEncoder setBuffer:_simulationParams offset:0 atIndex:AAPLComputeBufferIndexParams];

    [computeEncoder setThreadgroupMemoryLength:_threadgroupMemoryLength atIndex:0];

    [computeEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];

    [computeEncoder endEncoding];

    // Swap indices to use data generated this frame at _newBufferIndex to generate data for the
    // next frame and write it to the buffer at _oldBufferIndex
    uint8_t tmpIndex = _oldBufferIndex;
    _oldBufferIndex = _newBufferIndex;
    _newBufferIndex = tmpIndex;

    [commandBuffer popDebugGroup];

    _simulationTime += _config->simInterval;

    return _positions[_newBufferIndex];
}

@end
