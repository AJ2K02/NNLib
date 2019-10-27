
#ifndef NNL_UTILS_HPP
#define NNL_UTILS_HPP

#include "config.h"
#include <random>

#define LINEARIZE_INDEX_2D(y, x, x_size) ((y) * (x_size) + (x))
#define LINEARIZE_INDEX_3D(z, y, x, xy_size, x_size) ((z) * (xy_size) + (y) * (x_size) + (x))

#define SQR(x) ((x) * (x))

namespace nnl {

//Genere un nombre aleatoire selon une distribution normale
    Real randn(const double mean, const double stddev) {
        static thread_local auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static thread_local std::mt19937 gen(seed);
        std::normal_distribution<> distr{mean, stddev};
        return distr(gen);
    }

//Genere un nombre aleatoire selon une distribution uniforme
    Real uniform(const double lower_bound, const double higher_bound) {
        static thread_local auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static thread_local std::mt19937 gen(seed);
        std::uniform_real_distribution<> distr{lower_bound, higher_bound};
        return distr(gen);
    }

    Real mean(const RMMat <Real>& m) {
        return m.array().sum() / Real(m.rows() * m.cols());
    }

    Real stddev(const RMMat <Real>& m, const Real mean = 0) {
        return sqrt((m.array() - mean).square().sum() / Real(m.rows() * m.cols()));
    }

//Verifie que 0 <= x < b
//Utilise le fait que static_cast<unsigned>(-N) = UNSIGNED_MAX - N
//si -N est negatif pour ne faire qu'une comparaison
    bool ge_zero_lt_b(const int x, const unsigned b) {
        return static_cast<unsigned>(x) < b;
    }

} // namespace nnl
#endif //NNL_UTILS_HPP
