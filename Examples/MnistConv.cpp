//
// Created by Arthur on 14.09.2019.
//

#include <iostream>

#include "../Datasets/Mnist.hpp"
#include "../Dataset.hpp"
#include "../NeuralNetwork.hpp"
#include "../Tensor.hpp"
#include "../NeuralNetwork.hpp"

using namespace nnl;

int main() {
    Eigen::setNbThreads(1); //choix du nombres de threads a utiliser
    // Hyper-parametres
    constexpr unsigned DATA_SIZE = 60000;
    constexpr unsigned BATCH_SIZE = 100;
    constexpr unsigned EPOCHS = 1;

    // Chargement des inputs / labels
    Dataset dset = MNIST::load_data("C:/Users/Arthur/Desktop/CPP/Project1/Datasets/MNIST/raw");
    dset.add_labels_transforms(ToOneHot(10), Batch(BATCH_SIZE));
    dset.apply_labels_transforms(true, false);

    dset.add_data_transforms(Divide(255), Batch(BATCH_SIZE));
    dset.apply_data_transforms();

    // Creation du reseau neuronal
    NeuralNetwork nn;
    nn << Conv2d<Relu>(28, 1, 5, 20, 2, 0)
       << Conv2d<Relu>(12, 20, 5, 50, 2, 0)
       << FullyConnected<Relu>(4*4*50, 500)
       << FullyConnected<Sigmoid>(500, 10);
    nn.set_global_learning_rate(1e-3);

    // Apprentissage
    nn.train(dset, EPOCHS);

    // Test final :
    RMMat<Real> predictions;
    double correct = 0;
    unsigned total = 0;
    RMMat<Real>::Index pred;
    for (unsigned b = 0; b < 20; ++b) {
        predictions = nn(dset.test_data.as_matrix(10000, 784).block(500*b, 0, 500, 784));
        for (unsigned i = 0; i < predictions.rows(); ++i) {
            predictions.row(i).maxCoeff(&pred);
            if (pred == dset.test_labels.as_matrix().block(500*b, 0, 500, 1)(i, 0))
                ++correct;
            ++total;
        }
    }
    std::cout << "Accuracy on unseen data : " << 100.0 * correct / total << "%" << std::endl;
    return 0;
}

