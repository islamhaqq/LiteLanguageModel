#include <iostream>

#include "LLM.h"

int main()
{
    constexpr int numEpochs = 10;
    constexpr float learningRate = 0.01f;

    // Setup: Define model, allocate memory, initialize data, etc.
    DenseLayer denseLayer(768, 2048); // Layer

    // Create input tensor
    Tensor input({1, 768}); // Batch size 1, 768 features
    input.allocateMemoryOnDevice();

    // Create output tensor
    Tensor output({1, 2048});
    output.allocateMemoryOnDevice();

    // Create gradient tensors
    Tensor gradOutput({1, 2048}); // Assuming some gradient from the next layer or loss derivative
    gradOutput.allocateMemoryOnDevice();
    Tensor gradInput({1, 768}); // Gradient of the input tensor
    gradInput.allocateMemoryOnDevice();

    // Data initialization
    std::vector<float> inputData(768, 1.0f);
    std::vector<float> gradOutputData(2048, 0.1f);

    input.copyDataToDevice(inputData); // Fill 'inputData' with example data
    gradOutput.copyDataToDevice(gradOutputData);

    // Training loop for one iteration
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        denseLayer.forward(input, output);
        denseLayer.backward(input, gradInput, gradOutput);
        // @TODO: Update parameters (would involve applying gradients to weights and biases)
        // denseLayer.updateParameters(learningRate);
        std::cout << "Epoch " << epoch << " complete.\n";
    }

    return 0;
}
