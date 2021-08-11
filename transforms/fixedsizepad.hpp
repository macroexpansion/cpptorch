#ifndef FIXEDSIZEPAD_HPP
#define FIXEDSIZEPAD_HPP

#include <torch/torch.h>

struct FixedSizePaddingImpl : torch::nn::Module {
    int max_height, max_width;
    
    FixedSizePaddingImpl(int max_width, int max_height) 
        : max_width(max_width), max_height(max_height) {}

    torch::Tensor forward(torch::Tensor tensor) {
        namespace F = torch::nn::functional;
        auto height = tensor.size(2);
        auto width = tensor.size(3);
        int left_pad = 0;
        int right_pad = 0;
        int top_pad = 0;
        int bot_pad = 0;

        if (height < max_height) {
            int diff = max_height - height;
            top_pad = diff / 2;
            bot_pad = diff % 2 == 0 ? diff / 2 : diff / 2 + 1;
        }
        if (width < max_width) {
            int diff = max_width - width;
            left_pad = diff / 2;
            right_pad = diff % 2 == 0 ? diff / 2 : diff / 2 + 1;
        }
        tensor = F::pad(tensor, F::PadFuncOptions({left_pad, right_pad, top_pad, bot_pad})
            .mode(torch::kConstant)
            .value(0));
        return tensor;
    }
};

TORCH_MODULE(FixedSizePadding);

#endif