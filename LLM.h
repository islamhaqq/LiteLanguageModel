#ifndef LLM_H
#define LLM_H

#include "TensorLite.h"

/** A dense layer with a linear transformation */
class DenseLayer : public Operation {
public:
    DenseLayer(int inputSize, int outputSize);
    void forward(const Tensor& input, Tensor& output) override;
    void backward(const Tensor& input, Tensor& gradInput, const Tensor& gradOutput) override;

    int inputSize;
    int outputSize;
    Tensor* weights;
    Tensor* bias;
};

#endif //LLM_H
