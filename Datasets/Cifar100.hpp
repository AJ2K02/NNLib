//
// Created by Arthur on 25.10.2019.
//

#ifndef NNL_CIFAR100_HPP
#define NNL_CIFAR100_HPP


#include "../config.h"
#include "../Dataset.hpp"
#include "../Tensor.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

namespace nnl {

    struct CIFAR100 {
        static void read_file(const std::string& path, Real* out_data, Real* out_labels, const unsigned im_nb);

        static Dataset load_data(const std::string& _path,
                                 const ImageDataStorage ids = NonLinearizedImages);
    };


    void CIFAR100::read_file(const std::string& path, Real* out_data, Real* out_labels, const unsigned im_nb) {
        constexpr unsigned IMG_SIDE = 32;
        constexpr unsigned IMG_CHAN = 3;
        constexpr unsigned CLASSES = 100;
        std::ifstream file;
        file.open(path, std::ios::binary);
        if (file.is_open()) {
            std::unique_ptr<char[]> buf = std::make_unique<char[]>(3073);
            for (unsigned i = 0; i < im_nb; ++i) {
                // Lecture de 3073 bytes dans le buffer
                // buf[0] -> etiquette "coarse" (ne nous interesse pas)
                // buf[1] -> etiquette precise (les 100 classes)
                // buf[2:3074] -> image
                file.read(buf.get(), SQR(IMG_SIDE) * IMG_CHAN + 1);
                out_labels[i] = buf[1];
                for (unsigned j = 0; j < SQR(IMG_SIDE) * IMG_CHAN; ++j) {
                    out_data[i * 3072 + j] = uint8_t(buf[2 + j]);
                }
            }
        }
        //return read;
    }

    Dataset CIFAR100::load_data(const std::string& _path, const ImageDataStorage ids) {
        constexpr unsigned IMG_PER_FILE = 50000;
        const unsigned IMG_NB = 50000;
        constexpr unsigned IMG_CHAN = 3;
        constexpr unsigned IMG_SIDE = 32;
        constexpr unsigned DATA_PER_FILE = IMG_PER_FILE * IMG_CHAN * SQR(IMG_SIDE);
        constexpr unsigned TEST_NB = 10000;

        // Creation du dataset avec des Tensors crees mais vides

        Dataset ds{
                Tensor<Real>(IMG_NB),
                Tensor<Real>(IMG_NB, IMG_CHAN, IMG_SIDE, IMG_SIDE),
                Tensor<Real>(TEST_NB),
                Tensor<Real>(TEST_NB, IMG_CHAN, IMG_SIDE, IMG_SIDE),
        };

        if (ids == LinearizedImages) {
            ds.training_data.reshape_in_place_no_return(IMG_NB, IMG_CHAN * SQR(IMG_SIDE));
            ds.test_data.reshape_in_place_no_return(TEST_NB, IMG_CHAN * SQR(IMG_SIDE));
        }
        // Donnees d'entrainement
        read_file(_path + "/train.bin", ds.training_data.data(), ds.training_labels.data(), IMG_NB);

        // Ajout des donnees de test
        read_file(_path + "/test.bin", ds.test_data.data(), ds.test_labels.data(), TEST_NB);

        return ds;
    }

} // namespace nnl

#endif //NNL_CIFAR100_HPP
