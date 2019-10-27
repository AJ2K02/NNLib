#ifndef NNL_MNIST_HPP
#define NNL_MNIST_HPP

#include "../config.h"
#include "../Dataset.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../NeuralNetwork.hpp"

namespace nnl {

/*
 * Fonction utilisee pour interpreter les nombres au debut des fichiers
 * MNIST : les ints ont les bytes de poids faible a gauche, il faut donc
 * inverser les bytes. Les entiers font 32 bits / 4 bytes.
 */
    inline int reverse_int(const int32_t i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((int32_t) ch1 << 24) + ((int32_t) ch2 << 16) + ((int32_t) ch3 << 8) + ch4;
    }


    struct MNIST {
        inline static Tensor <Real>
        read_images(const std::string& path_, const unsigned im_nb, const bool training = true,
                    const ImageDataStorage ids = NonLinearizedImages);

        inline static Tensor <Real>
        read_labels(const std::string& path_, const unsigned im_nb, const bool training = true);

        inline static Dataset load_data(const std::string& path_, const ImageDataStorage ids = NonLinearizedImages);
    };


    inline Tensor <Real> MNIST::read_images(const std::string& path_, const unsigned im_nb, const bool training,
                                            const ImageDataStorage ids) {
        std::string path = path_;
        if (training) path += "/train-images-idx3-ubyte";
        else path += "/t10k-images-idx3-ubyte";
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0, n_cols = 0;

            // Lire le nombre magique et le nombre d'images
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = reverse_int(magic_number);
            file.read((char*) &number_of_images, sizeof(number_of_images));
            number_of_images = reverse_int(number_of_images);
            number_of_images = (number_of_images < im_nb) ? number_of_images : im_nb;

            // Lire le nombre de lignes/colonnes par image
            file.read((char*) &n_rows, sizeof(n_rows));
            n_rows = reverse_int(n_rows);
            file.read((char*) &n_cols, sizeof(n_cols));
            n_cols = reverse_int(n_cols);
            assert(n_rows == n_cols && n_rows == 28 && "Incorrect number of rows/columns per image");

            // Lire les pixels et les placer dans une matrice
            const unsigned pixels = n_rows * n_cols; // Pixels par image, normalement 784
            RMMat<Real> data(number_of_images, pixels);
            for (int i = 0; i < number_of_images; ++i) {
                unsigned char im[pixels];
                file.read((char*) im, sizeof(im[0]) * pixels);
                for (unsigned c = 0; c < pixels; ++c) {
                    data(i, c) = im[c];
                }
            }
            return (ids == NonLinearizedImages) ?
                   Tensor<Real>(data, number_of_images, 1, 28, 28) :
                   Tensor<Real>(data, number_of_images, 784);
        } else {
            std::cout << "Couldn't open file with path : " << path << std::endl;
        }
    }


    inline Tensor <Real> MNIST::read_labels(const std::string& path_, const unsigned im_nb, const bool training) {
        RMMat<Real> labels;
        std::string path = path_;
        if (training) path += "/train-labels-idx1-ubyte";
        else path += "/t10k-labels-idx1-ubyte";

        std::ifstream file(path, std::ios::binary);
        int number_of_images = 0;
        if (file.is_open()) {
            int magic_number = 0;
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = reverse_int(magic_number);
            file.read((char*) &number_of_images, sizeof(number_of_images));
            number_of_images = reverse_int(number_of_images);
            number_of_images = (number_of_images < im_nb) ? number_of_images : im_nb;
            labels = RMMat<Real>::Zero(number_of_images, 1);

            for (unsigned i = 0; i < number_of_images; ++i) {
                unsigned char label = 0;
                file.read((char*) &label, 1);
                labels(i, 0) = label;
            }
        } else
            std::cout << "Could not open file (labels)\n";
        return Tensor<Real>(labels, number_of_images, 1);
    }

    inline Dataset MNIST::load_data(const std::string& path_, const ImageDataStorage ids) {
        Dataset ds;
        ds.training_data = read_images(path_, std::numeric_limits<unsigned>::max(), true, ids);
        ds.training_labels = read_labels(path_, std::numeric_limits<unsigned>::max(), true);
        ds.test_data = read_images(path_, std::numeric_limits<unsigned>::max(), false, ids);
        ds.test_labels = read_labels(path_, std::numeric_limits<unsigned>::max(), false);
        return ds;
    }

    inline double test_accuracy(const Matrix& inputs,
                                const Matrix& labels,
                                NeuralNetwork& nn) {
        Matrix guesses = nn(inputs);
        Eigen::Index max_idx, label;
        unsigned correct = 0;
        for (unsigned i = 0; i < guesses.rows(); ++i) {
            guesses.row(i).maxCoeff(&max_idx);
            labels.row(i).maxCoeff(&label);
            if (max_idx == label)
                ++correct;
        }
        return (double) correct / labels.rows();
    }

} // namespace nnl

#endif //NNL_MNIST_HPP