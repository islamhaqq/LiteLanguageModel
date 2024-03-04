#include "LLM.h"

int main()
{
    // Define model
    DenseLayer denseLayer(768, 2048);

    // Create input tensor
    Tensor input({1, 768}); // Batch size 1, 768 features
    input.allocateMemoryOnDevice();
    // Fill 'inputData' with example data
    std::vector<float> inputData(768, 1.0f);
    input.copyDataToDevice(inputData);

    // Create output tensor
    Tensor output({1, 2048});
    output.allocateMemoryOnDevice();

    // Forward pass
    denseLayer.forward(input, output);

    return 0;
}
