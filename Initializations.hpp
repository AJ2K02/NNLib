
#ifndef NNL_INITIALIZATIONS_HPP
#define NNL_INITIALIZATIONS_HPP

#include "config.h"
#include "utils.hpp"

#include <cmath>

namespace nnl {


    class KaimingUniform {
    public:
        static void initialize_weight(RMMat <Real>& w, const unsigned fan_in, const unsigned fan_out);

        static void initialize_bias(RMMat <Real>& b, const unsigned fan_in, const unsigned fan_out);
    };


    void KaimingUniform::initialize_weight(RMMat <Real>& w, const unsigned fan_in, const unsigned fan_out) {
        double bound = sqrt(1.0 / fan_in);
        auto rand_gen = std::bind(uniform, -bound, bound);
        for (unsigned i = 0; i < w.rows() * w.cols(); ++i)
            w.data()[i] = rand_gen();
    }

    void KaimingUniform::initialize_bias(RMMat <Real>& b, const unsigned fan_in, const unsigned fan_out) {
        double bound = sqrt(1.0 / fan_in);
        auto rand_gen = std::bind(uniform, -bound, bound);
        for (unsigned i = 0; i < b.rows() * b.cols(); ++i)
            b.data()[i] = rand_gen();
    }

} // namespace nnl

#endif //NNL_INITIALIZATIONS_HPP
