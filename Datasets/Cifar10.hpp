
#ifndef NNL_CIFAR10_HPP
#define NNL_CIFAR10_HPP

#include "../config.h"
#include "../Dataset.hpp"
#include "../Tensor.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

namespace nnl {

    struct _DataWithLabel_Temp {
        Tensor <Real> data;
        Tensor <Real> labels;
    };


    struct CIFAR10 {
        static void read_file(const std::string& path, Real* out_data, Real* out_labels);

        static Dataset load_data(const std::string& _path, const unsigned nb_of_files = 5,
                                 const ImageDataStorage ids = NonLinearizedImages);
    };


    void CIFAR10::read_file(const std::string& path, Real* out_data, Real* out_labels) {
        constexpr unsigned IMG_PER_FILE = 10000;
        constexpr unsigned IMG_SIDE = 32;
        constexpr unsigned IMG_CHAN = 3;
        constexpr unsigned CLASSES = 10;
        std::ifstream file;
        file.open(path, std::ios::binary);
        if (file.is_open()) {
            std::unique_ptr<char[]> buf = std::make_unique<char[]>(3073);
            for (unsigned i = 0; i < IMG_PER_FILE; ++i) {
                file.read(buf.get(), SQR(IMG_SIDE) * IMG_CHAN + 1);
                //read.labels(i).item() = buf[SQR(IMG_SIDE) * IMG_CHAN];
                out_labels[i] = buf[0];
                for (unsigned j = 0; j < SQR(IMG_SIDE) * IMG_CHAN; ++j) {
                    //read.data(i).data()[j] = buf[j];
                    out_data[i * 3072 + j] = uint8_t(buf[1 + j]);
                }
            }
        }
        //return read;
    }

    Dataset CIFAR10::load_data(const std::string& _path, const unsigned nb_of_files, const ImageDataStorage ids) {
        constexpr unsigned IMG_PER_FILE = 10000;
        const unsigned IMG_NB = 10000 * nb_of_files;
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

        for (unsigned i = 1; i <= nb_of_files; ++i)
            read_file(_path + "/data_batch_" + std::to_string(i) + ".bin", ds.training_data(10000 * (i - 1)).data(),
                      ds.training_labels(10000 * (i - 1)).data());

        // Ajout des donnees de test
        read_file(_path + "/test_batch.bin", ds.test_data.data(), ds.test_labels.data());

        return ds;
    }

} // namespace nnl

#endif //NNL_CIFAR10_HPP
