
#ifndef NNL_LOSS_HPP
#define NNL_LOSS_HPP

#include "config.h"

namespace nnl {

//Mean Squared Error Loss
    class MSELoss {
    public:
        static Real forward(const RMMat <Real>& pred, const RMMat <Real>& labels) {
            return (pred - labels).array().square().sum();
        }

        static RMMat <Real> backward(const RMMat <Real>& pred, const RMMat <Real>& labels) {
            return 2 * (pred - labels);
        }
    };

} // namespace nnl

#endif //NNL_LOSS_HPP
