#include <iostream>

#include "../Datasets/Mnist.hpp"
#include "../NeuralNetwork.hpp"
#include "../Tensor.hpp"

using namespace nnl;

int main()
{
    Eigen::setNbThreads(8); //choix du nombres de threads a utiliser
    // Hyper-parametres
    constexpr unsigned DATA_SIZE = 60000;
    constexpr unsigned BATCH_SIZE = 200;
    constexpr unsigned EPOCHS = 3;

    // Chargement des inputs / labels
    RMMat<Real> raw_inputs = read_mnist_images(DATA_SIZE);
    RMMat<Real> raw_labels = to_one_hot(read_mnist_labels(DATA_SIZE));
    Tensor<Real> inputs(raw_inputs, raw_inputs.rows()/BATCH_SIZE, BATCH_SIZE, raw_inputs.cols());
    Tensor<Real> labels(raw_labels, raw_labels.rows()/BATCH_SIZE, BATCH_SIZE, raw_labels.cols());

    // Chargement du dataset de test
    RMMat<Real> test_in = read_mnist_images(10000, false);
    RMMat<Real> test_labels = read_mnist_labels(10000, false);

    // Creation du reseau neuronal
    NeuralNetwork nn;
    nn << FullyConnected<Relu>(784, 100)
       << FullyConnected<Relu>(100, 100)
       << FullyConnected<Sigmoid>(100, 10);
    nn.set_global_learning_rate(1e-3);

    // Apprentissage
    nn.batched_train(inputs, labels, EPOCHS);

    // Test final :
    RMMat<Real> predictions = nn(test_in);
    double correct = 0;
    unsigned total = 0;
    RMMat<Real>::Index pred;
    for (unsigned i = 0; i < predictions.rows(); ++i) {
        predictions.row(i).maxCoeff(&pred);
        if (pred == test_labels(i, 0))
            ++correct;
        ++total;
    }
    std::cout << "Accuracy on unseen data : " << 100.0 * correct / total << "%" << std::endl;

    return 0;
}