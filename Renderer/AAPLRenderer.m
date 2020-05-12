/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of renderer class which performs Metal setup and per frame rendering
*/
@import simd;
@import MetalKit;

#import "AAPLRenderer.h"
#import "AAPLMathUtilities.h"

// Include header shared between C code here, which executes Metal API commands, and .metal files
#import "AAPLShaderTypes.h"

// The point size (in pixels) of rendered bodied
static const float AAPLBodyPointSize = 15;

// Size of gaussian map to create rounded smooth points
static const NSUInteger AAPLGaussianMapSize = 64;

// Main class performing the rendering
@implementation AAPLRenderer
{
    id<MTLTexture> _gaussianMap;

    id<MTLCommandQueue> _commandQueue;

    id<MTLBuffer> _colors;

    // Metal objects
    id<MTLBuffer> _positionsBuffer;
    id<MTLBuffer> _dynamicUniformBuffer;
    id<MTLRenderPipelineState> _renderPipeline;
    id<MTLDepthStencilState> _depthState;

    // Projection matrix calculated as a function of view size
    matrix_float4x4 _projectionMatrix;

    float _renderScale;
}

/// Initialize with the MetalKit view with the Metal device used to render.  This MetalKit view
/// object will also be used to set the pixelFormat and other properties of the drawable
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if(self)
    {
        _device = mtkView.device;

        [self loadMetal:mtkView];
        [self generateGaussianMap];
    }

    return self;
}

/// Generates a texture to make rounded points for particles
-(void) generateGaussianMap
{
    NSError *error;

    MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];

    textureDescriptor.textureType = MTLTextureType2D;
    textureDescriptor.pixelFormat = MTLPixelFormatR8Unorm;
    textureDescriptor.width = AAPLGaussianMapSize;
    textureDescriptor.height = AAPLGaussianMapSize;
    textureDescriptor.mipmapLevelCount = 1;
    textureDescriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    textureDescriptor.usage = MTLTextureUsageShaderRead;

    _gaussianMap = [_device newTextureWithDescriptor:textureDescriptor];

    // Calculate the size of a RGBA8Unorm texture's data and allocate system memory buffer
    // used to fill the texture's memory
    NSUInteger dataSize = textureDescriptor.width  * textureDescriptor.height  * sizeof(uint8_t);

    const vector_float2 nDelta = simd_make_float2(
        2.0 / (float)textureDescriptor.width,
        2.0 / (float)textureDescriptor.height
    );

    uint8_t* texelData = (uint8_t*) malloc(dataSize);

    int i = 0;

    // Procedurally generate data to fill the texture's buffer
    for(uint32_t y = 0; y < textureDescriptor.height; y++)
    {
        float const sNormY = -1.0 + y * nDelta.y;

        for(uint32_t x = 0; x < textureDescriptor.width; x++)
        {
            float const sNormX = -1.0 + x * nDelta.x;

            vector_float2 const sNormVector = simd_make_float2(sNormX, sNormY);
            float const h = MIN(1.0f, vector_length(sNormVector));

            // Hermite interpolation where u = {1, 0} and v = {0, 0}
            float const color = (2.0f * h - 3.0f) * h * h + 1.0f;

            texelData[i] = 0xFF * color;

            i++;
        }
    }

    MTLRegion region = {{ 0, 0, 0 }, {textureDescriptor.width, textureDescriptor.height, 1}};

    [_gaussianMap replaceRegion:region
                    mipmapLevel:0
                      withBytes:texelData
                    bytesPerRow:sizeof(uint8_t) * textureDescriptor.width];

    if(!_gaussianMap || error)
    {
        NSLog(@"Error creating gaussian map: %@", error.localizedDescription);
    }

    _gaussianMap.label = @"Gaussian Map";

    free(texelData);
}

/// Create the Metal render state objects including shaders and render state pipeline objects
- (void) loadMetal:(nonnull MTKView *)mtkView
{
    NSError *error = nil;

    // Load all the shader files with a .metal file extension in the project
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

    // Load the vertex function from the library
    id<MTLFunction> vertexFunction = [defaultLibrary newFunctionWithName:@"vertexShader"];

    // Load the fragment function from the library
    id<MTLFunction> fragmentFunction = [defaultLibrary newFunctionWithName:@"fragmentShader"];

    mtkView.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    mtkView.sampleCount = 1;

    {
        MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.label = @"RenderPipeline";
        pipelineDescriptor.sampleCount = mtkView.sampleCount;
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = fragmentFunction;
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;
        pipelineDescriptor.depthAttachmentPixelFormat = mtkView.depthStencilPixelFormat;
        pipelineDescriptor.stencilAttachmentPixelFormat = mtkView.depthStencilPixelFormat;
        pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor  = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;

        _renderPipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
        if (!_renderPipeline)
        {
            NSLog(@"Failed to create render pipeline state, error %@", error);
        }
    }

    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStateDesc.depthWriteEnabled = YES;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];

    // Indicate shared storage so that both the  CPU can access the buffers
    const MTLResourceOptions storageMode = MTLResourceStorageModeShared;

    _dynamicUniformBuffer = [_device newBufferWithLength:sizeof(AAPLUniforms)
                                              options:storageMode];

    _dynamicUniformBuffer.label = @"UniformBuffer";

     _commandQueue = [_device newCommandQueue];
}

/// Set the number of bodies to render and create a buffer to color each body
- (void)setNumRenderBodies:(NSUInteger)numBodies
{
    if(!_colors || ((_colors.length / sizeof(vector_uchar4)) < numBodies))
    {
        // If the number of colors stored is less than the number of bodies, recreate the color buffer

        NSUInteger bufferSize = numBodies * sizeof(vector_uchar4);

        _colors = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeManaged];

        _colors.label = @"Colors";

        vector_uchar4 *colors = (vector_uchar4 *) _colors.contents;

        for(int i = 0; i < numBodies; i++)
        {
            vector_float3 color = generate_random_vector(0, 1);

            colors[i].x = 0xFF * color.x;
            colors[i].y = 0xFF * color.y;
            colors[i].z = 0xFF * color.z;
            colors[i].w = 0xFF;
        }
        NSRange fullDataRange = NSMakeRange(0, bufferSize);

        [_colors didModifyRange:fullDataRange];
    }
}

/// Update the projection matrix with a new render buffer size
- (void)updateProjectionMatrixWithSize:(CGSize)size
{
    // React to resize of the draw rect.  In particular update the perspective matrix.
    // Update the aspect ratio and projection matrix since the view orientation or size has changed
    const float aspect = (float)size.height / size.width;
    const float left   = _renderScale;
    const float right  = -_renderScale;
    const float bottom = _renderScale * aspect;
    const float top    = -_renderScale * aspect;
    const float near   = 5000;
    const float far    = -5000;

    _projectionMatrix = matrix_ortho_left_hand(left, right, bottom, top, near, far);
}

/// Set the scale factor and parameters to create the projection matrix
- (void)setRenderScale:(float)renderScale withDrawableSize:(CGSize)size
{
    _renderScale = renderScale;

    [self updateProjectionMatrixWithSize:size];
}

/// Update the projection matrix with a new drawable size
- (void)drawableSizeWillChange:(CGSize)size
{
//    [self updateProjectionMatrixWithSize:size];
}

/// Update any render state (including updating dynamically changing Metal buffers)
- (void)updateState
{
    AAPLUniforms *contents = (AAPLUniforms *)_dynamicUniformBuffer.contents;

    contents->pointSize = AAPLBodyPointSize;
    contents->mvpMatrix = _projectionMatrix;
}

/// Called to provide positions data to be rendered on the next frame
- (void)providePositionData:(NSData *)data
{
    // Synchronize since positions buffer will be used on another thread
    @synchronized(self)
    {
        // Cast from 'const void *' to 'void *' which is okay in this case since updateData was
        // created with -[NSData initWithBytesNoCopy:length:deallocator:] and underlying memory was
        // allocated with vm_allocate
        void *vmAllocatedAddress = (void *)data.bytes;

        // Create a MTLBuffer with out copying the data
        id<MTLBuffer> positionsBuffer = [_device newBufferWithBytesNoCopy:vmAllocatedAddress
                                                                   length:data.length
                                                                  options:MTLResourceStorageModeManaged
                                                              deallocator:nil];

        positionsBuffer.label = @"Provided Positions";

        [positionsBuffer didModifyRange:NSMakeRange(0, data.length)];

        _positionsBuffer = positionsBuffer;
    }
}

/// Draw particles at the supplied positions using the given command buffer to the given view
- (void)drawWithCommandBuffer:(nonnull id<MTLCommandBuffer>)commandBuffer
              positionsBuffer:(nonnull id<MTLBuffer>)positionsBuffer
                    numBodies:(NSUInteger)numBodies
                       inView:(nonnull MTKView *)view
{
    [commandBuffer pushDebugGroup:@"Draw Simulation Data"];

    [self setNumRenderBodies:numBodies];

    [self updateState];

    // Obtain a renderPassDescriptor generated from the view's drawable textures
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;

    // If a renderPassDescriptor has been obtained, render to the drawable, otherwise skip
    // any rendering this frame because there is no drawable to draw to
    if(renderPassDescriptor != nil)
    {
        id<MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

        renderEncoder.label = @"Render Commands";

        [renderEncoder setRenderPipelineState:_renderPipeline];

        if(positionsBuffer)
        {
            // Synchronize since positions buffer may be created on another thread
            @synchronized(self)
            {
                [renderEncoder setVertexBuffer:positionsBuffer
                                        offset:0 atIndex:AAPLRenderBufferIndexPositions];
            }

            [renderEncoder setVertexBuffer:_colors
                                    offset:0 atIndex:AAPLRenderBufferIndexColors];

            [renderEncoder setVertexBuffer:_dynamicUniformBuffer
                                    offset:0 atIndex:AAPLRenderBufferIndexUniforms];

            [renderEncoder setFragmentTexture:_gaussianMap atIndex:AAPLTextureIndexColorMap];

            [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                              vertexStart:0
                              vertexCount:numBodies
                            instanceCount:1];
        }

        [renderEncoder endEncoding];

        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer popDebugGroup];
}

@end
