#include <iostream>
#include "NNLib.hpp"

using namespace nnl;

double test(NeuralNetwork& nn, Tensor<Real>& data, Tensor<Real>& labels) {
    RMMat<Real> predictions;
    double correct = 0;
    unsigned total = 0;
    RMMat<Real>::Index pred;
    for (unsigned b = 0; b < 100; ++b) {
        predictions = nn(data.as_matrix(10000, 3072).block(100*b, 0, 100, 3072));
        for (unsigned i = 0; i < predictions.rows(); ++i) {
            predictions.row(i).maxCoeff(&pred);
            if (pred == labels.as_matrix(10000, 1).block(100*b, 0, 100, 1)(i, 0))
                ++correct;
            ++total;
        }
    }
    std::cout << "Accuracy : " << 100.0 * correct / total << "%." << std::endl;
}

int main()
{
    std::cout << "Starting to load data.\n";
    Dataset d = CIFAR10::load_data("C:\\Users\\Arthur\\Desktop\\CPP\\Project1\\Datasets\\CIFAR-10", 5, NonLinearizedImages);
    auto original_train_labels = d.training_labels;
    d.add_data_transforms(Divide(255.0), Batch(100));
    d.add_labels_transforms(ToOneHot(10), Batch(100));
    d.apply_data_transforms(true, true);
    d.apply_labels_transforms(true, false);
    std::cout << "Data loaded and transformed.\n";

    // Creation du reseau neuronal
    NeuralNetwork nn;
    nn << Conv2d<LeakyRelu>(32, 3, 3, 20, 1, 1).init<KaimingUniform>() //out : 32
       << Conv2d<LeakyRelu>(32, 20, 5, 50, 3, 0).init<KaimingUniform>() //out : 10
       << Conv2d<LeakyRelu>(10, 50, 3, 64, 2, 1).init<KaimingUniform>() //out : 5
       << FullyConnected<LeakyRelu>(5*5*64, 1000)
       << FullyConnected<LeakyRelu>(1000, 300)
       << FullyConnected<Sigmoid>(300, 10);
    nn.set_global_learning_rate(3e-3);
    std::cout << "NeuralNet is valid : " << nn.is_valid() << std::endl;

    for (unsigned i = 0; i < 10; ++i) {
        std::cout << "Step #" << i << '\n';
        // Apprentissage
        nn.train(d, 2);

        // Sauvegarde
        std::string filename("Save_");
        filename += std::to_string(i) + ".txt";
        nn.save(filename);

        // Test
        std::cout << "\nTests pour le fichier #" << i << ".\n";
        test(nn, d.training_data, original_train_labels);
        test(nn, d.test_data, d.test_labels);
    }
    std::cout << std::flush;

    return 0;

}