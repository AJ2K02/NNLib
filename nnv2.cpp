#include <iostream>
#include "NNLib.hpp"

using namespace nnl;

double test(NeuralNetwork& nn, Tensor<Real>& data, Tensor<Real>& labels) {
    const std::size_t NB_VEC = data.get_dimensions()[0] * data.get_dimensions()[1];
    const std::size_t BATCH_SIZE = data.get_dimensions()[1];

    unsigned correct = 0, total = 0;
    Tensor<Real> predictions;
    for (unsigned i = 0; i < NB_VEC/BATCH_SIZE;++i) {
        auto temp = data(i);
        predictions = nn(temp);
        for (unsigned j = 0; j < predictions.get_dimensions()[0]; ++j) {
            if (predictions(j).argmax() == labels(i, j).argmax())
                ++correct;
            ++total;
        }
    }
    std::cout << "Accuracy : " << 100.0 * correct / total << "%." << std::endl;
}

int main()
{
    std::cout << "Starting to load data.\n";
    Dataset d = CIFAR10::load_data("../Datasets/CIFAR-10", 1, NonLinearizedImages);
    auto original_train_labels = d.training_labels;
    d.add_data_transforms(Divide(255.0), Batch(100));
    d.add_labels_transforms(ToOneHot(10), Batch(100));
    d.apply_data_transforms(true, true);
    d.apply_labels_transforms(true, true);
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
        //nn.train(d, 2);

        // Sauvegarde
        //std::string filename("Save_");
        //filename += std::to_string(i) + ".txt";
        //nn.save(filename);

        // Test
        std::cout << "\nTests pour le fichier #" << i << ".\n";
        test(nn, d.training_data, d.training_labels);
        test(nn, d.test_data, d.test_labels);
    }
    std::cout << std::flush;
    return 0;

}
