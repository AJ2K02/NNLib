
/*
 * Ce header contient les differentes fonctions d'activations utilisables.
 * On peut facilement en rajouter en creeant une classe suivant ce modèle :
 *
 *      class MonActivation {
 *      public:
 *          static std::string get_name() { ... } // retourne le nom de la fonction
 *          static Real forward(const Real x) { ... }    // ici on effectue le calcul de la fonction
 *          static Real backward(const Real x) { ... } // derivee
 *      };
 *
 * Fonctions actuellement implementees : //TODO en rajouter!!!!
 * - relu
 * - identite (Linear)
 * - tanh
 * - sigmoid
 * - LeakyRelu
 */
#ifndef NNL_ACTIVATIONS_HPP
#define NNL_ACTIVATIONS_HPP

#include <cmath>
#include <string>

#include "config.h"

namespace nnl {
    class Relu {
    public:
        static Real forward(const Real x) {
            return x >= 0 ? x : 0;
        }

        static Real backward(const Real x) {
            return x >= 0 ? 1 : 0;
        }

        static std::string get_name() {
            return "Relu";
        }
    };


    class Linear { // fonction identite
    public:
        static Real forward(const Real x) {
            return x;
        }

        static Real backward(const Real x) {
            return 1;
        }

        static std::string get_name() {
            return "Linear";
        }
    };


    class Tanh {
    public:
        static Real forward(const Real x) {
            return tanh(x);
        }

        static Real backward(const Real x) {
            return 1.0 / (cosh(x) * cosh(x));
        }

        static std::string get_name() {
            return "Tanh";
        }
    };


    class Sigmoid {
    public:
        static Real forward(const Real x) {
            return 1 / (1 + exp(-x));
        }

        static Real backward(const Real x) {
            return forward(x) * (1 - forward(x));
        }

        static std::string get_name() {
            return "Sigmoid";
        }
    };


    class LeakyRelu {
    public:
        static Real forward(const Real x) {
            return x > 0 ? x : 0.2 * x;
        }

        static Real backward(const Real x) {
            return x > 0 ? 1 : 0.2;
        }

        static std::string get_name() {
            return "LeakyRelu";
        }
    };

} //namespace nnl

#endif //NNL_ACTIVATIONS_HPP
