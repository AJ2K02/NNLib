/*
 * Exemple ou on cree un reseau de neurone pour approximer une fonction mathematique f : R -> R.
 * Le dataset est constitue de nombres en input et de f(nombres) comme etiquettes.
 */


#include "../NeuralNetwork.hpp"
#include "../Tensor.hpp"

#include <cmath>
#include <iostream>

using namespace nnl;

int main()
{
    Eigen::setNbThreads(1); //choix du nombres de threads a utiliser
    constexpr unsigned EX = 1000;
    constexpr unsigned BS = 5;
    constexpr unsigned TEST_EX = 100;
    auto f = [](double& x) { x = std::cos(x); };
    // dataset d'entrainement pour une fonction f
    Tensor<double> inputs = random_tensor_d(BS, EX/BS, 1); inputs *= 6.28; //tensor random de 0 a 6.28 (2pi)
    Tensor<double> targets = inputs;
    targets.foreach(f);
    // dataset de test
    Tensor<double> test_in = random_tensor_d(1, TEST_EX, 1); test_in *= 6.28;
    Tensor<double> test_tar = test_in;
    test_tar.foreach(f);

    // entrainer NN
    NeuralNetwork nn;
    nn << FullyConnected<Relu>(1, 50)
       << FullyConnected<Relu>(50, 50)
       << FullyConnected<Linear>(50, 1);
    nn.set_global_learning_rate(1e-4);
    nn.batched_train(inputs, targets, 2000);

    //test du nn et affichage des predictions
    auto print_2t = [](const Tensor<Real>& t1, const Tensor<Real>& t2, const unsigned nb) {
        for (unsigned i = 0; i < nb; ++i)
            std::cout << t1.data()[i] << "   " << t2.data()[i] << std::endl;
    };
    auto pred = nn(test_in(0));
    std::cout << "TEST\nPREDICTIONS, EXPECTED : \n";
    print_2t(pred, test_tar, TEST_EX);

    return 0;
}

