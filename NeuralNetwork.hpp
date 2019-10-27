#ifndef NNL_NEURALNETWORK_HPP
#define NNL_NEURALNETWORK_HPP

#include "config.h"
//#include "Conv2d_with_tensors.hpp"
#include "Dataset.hpp"
#include "FullyConnected.hpp"
#include "Layer.hpp"
#include "Loss.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <math.h>
#include <memory>
#include <string>
#include <vector>

namespace nnl {


    class NeuralNetwork {
    public:
        //-* Constructeur par défaut.
        NeuralNetwork() = default;

        //-* Constructeur par copie par défaut
        NeuralNetwork(const NeuralNetwork& nn) = default;

        //-* Destructeur par défaut.
        ~NeuralNetwork() = default;

        //NeuralNetwork& operator<<(const FullyConnected& layer);
        //NeuralNetwork& operator<<(const Convolutional& layer);
        //-* Permet d'ajouter une couche au réseau neuronal.
        template<typename LayerType>
        inline NeuralNetwork& operator<<(const LayerType& layer);

        //-* Passe la matrice d'entrée %%inputs au réseau neuronal et retourne la matrice de sortie.
        inline Matrix operator()(const Matrix& inputs);

        //-* Passe le tenseur d'entrée %%inputs au réseau neuronal et retourne le tenseur de sortie.
        inline Tensor<Real> operator()(Tensor<Real>& inputs);

        /*-*
         * Vérifie la valditié de l'enchaînement des couches du réseau neuronal. Vérifie que le nombre
         * de valeurs de sorties de chaque couche correspond au nombre de valeurs d'entrées de la couche suivante.
        *-*/
        inline bool is_valid();

        //-* Défini le taux d'apprentissage de chaque couche avec la valeur %%lr.
        inline void set_global_learning_rate(const double lr);

        /*-*
         * Met à jour les paramètres de chaque couche
         * en utilisant les gradients (qui doivent avoir été calculés au préalable).
        *-*/
        inline void update_params();

        /*-*
         * Prend des entiers naturels en paramètres qui indiquent les couches à geler.
         * Geler une couche signifie que ses paramètres ne sont pas à mettre à jour lors de
         * la rétro-propagation.
         *-*/
        template<typename ...Args>
        inline void freeze(Args&& ...layers);

        /*-*
         * Prend des entiers naturels en paramètres qui indiquent les couches à dégeler.
         * Dégeler une couche signifie que ses paramètres sont à mettre à jour lors de
         * la rétro-propagation.
         *-*/
        template<typename ...Args>
        inline void unfreeze(Args&& ...layers);

        //-* Méthode effectuant la rétro-propagation.
        template<typename Loss = MSELoss>
        void backpropagation(const Matrix& inputs, const Matrix& outputs, const Matrix& labels);

        /*-*
         * Méthode permettant d'entraîner le réseau neuronal sur le tenseur d'entrée %%inputs
         * et ses étiquettes correspondantes %%labels. %%epochs indique le nombre d'époques
         * effectuées.
        *-*/
        template<typename Loss = MSELoss, typename T>
        inline void train(Tensor<T>& inputs, Tensor<T>& labels, unsigned epochs);

        /*-*
         * Méthode permettant d'entraîner le réseau neuronal sur le tenseur d'entrée %%inputs
         * et ses étiquettes correspondantes %%labels. %%epochs indique le nombre d'époques
         * effectuées.
         * Le paramètre %%lr_updater est un functor appelé à chaque itération, prenant en paramètre
         * le taux d'apprentissage actuel et le nombre d'itérations effectuées. Ce functor retourne
         * le nouveau taux d'apprentissage.
        *-*/
        template<typename Loss = MSELoss, typename T>
        inline void train(Tensor<T>& inputs, Tensor<T>& labels, unsigned epochs,
                          std::function<Real(const Real, const unsigned)>& lr_updater);

        /*-*
         * Méthode permettant d'entraîner le réseau neuronal sur le dataset %%dataset. %%epochs indique
         * le nombre d'époques effectuées.
        *-*/
        template<typename Loss = MSELoss>
        inline void train(Dataset& dataset, const unsigned epochs);

        /*-*
         * Méthode permettant d'entraîner le réseau neuronal sur le dataset %%dataset. %%epochs indique
         * le nombre d'époques effectuées.
         * Le paramètre %%lr_updater est un functor appelé à chaque itération, prenant en paramètre
         * le taux d'apprentissage actuel et le nombre d'itérations effectuées. Ce functor retourne
         * le nouveau taux d'apprentissage.
        *-*/
        template<typename Loss = MSELoss>
        inline void train(Dataset& dataset, const unsigned epochs,
                          std::function<Real(const Real, const unsigned)> lr_updater);

        //-* Enregistre le réseau neuronal dans un fichier dont le chemin d'accès est %%path.
        inline void save(const std::string& path) const;

        //-* Charge les paramètres du réseau neuronal depuis un fichier dont le chemin d'accès est %%path.
        inline void load(const std::string& path);

    private:
        std::vector<std::unique_ptr<Layer> > layers;
    };


//Methode pour ajouter une couche au reseau
    template<typename LayerType>
    inline NeuralNetwork& NeuralNetwork::operator<<(const LayerType& layer) {
        layers.push_back(std::make_unique<LayerType>(layer));
        return *this;
    }


    inline Matrix NeuralNetwork::operator()(const Matrix& inputs) {
        auto output = layers[0]->forward(inputs);
        for (unsigned i = 1; i < layers.size(); i++)
            output = layers[i]->forward(output);
        return output;
    }


    inline Tensor<Real> NeuralNetwork::operator()(Tensor<Real>& inputs) {
        std::vector<std::size_t> size = inputs.get_dimensions();
        auto inp_mat = inputs.as_matrix(size[0], std::accumulate(size.begin() + 1, size.end(), 1, std::multiplies<>()));
        auto output = layers[0]->forward(inp_mat);
        for (unsigned i = 1; i < layers.size(); i++)
            output = layers[i]->forward(output);
        return Tensor<Real>(output);
    }

    inline bool NeuralNetwork::is_valid() {
        bool valid = true;
        unsigned inputs, last_outputs(layers[0]->get_inputs_nb());
        unsigned layer_cnt = 0;
        for (auto& layer : layers) {
            inputs = layer->get_inputs_nb();
            if (inputs != last_outputs) {
                std::cerr << "Problem at layer " << layer_cnt << ". Outputs of previous layer = " << last_outputs
                          << "; Inputs of layer = " << inputs << std::endl;
                valid = false;
            }
            last_outputs = layer->get_outputs_nb();
            ++layer_cnt;
        }
        return valid;
    }

    inline void NeuralNetwork::set_global_learning_rate(const double lr) {
        for (std::unique_ptr<Layer>& layer : layers)
            layer->set_lr(lr);
    }


    inline void NeuralNetwork::update_params() {
        for (std::unique_ptr<Layer>& layer : layers)
            layer->update_param();
    }


    template<typename ...Args>
    inline void NeuralNetwork::freeze(Args&& ...layers_) {
        std::vector<unsigned> layers_v{static_cast<std::size_t>(layers_)...};
        for (unsigned layer : layers_v)
            layers[layer]->freeze(true);
    }

    template<typename ...Args>
    inline void NeuralNetwork::unfreeze(Args&& ...layers_) {
        std::vector<unsigned> layers_v{static_cast<std::size_t>(layers_)...};
        for (unsigned layer : layers_v)
            layers[layer]->freeze(false);
    }

    template<typename Loss>
    inline void NeuralNetwork::backpropagation(const Matrix& inputs, const Matrix& outputs, const Matrix& labels) {
        Matrix temp = Loss::backward(outputs, labels);
        if (layers.size() > 1) {
            //derniere couche faite d'abord,
            //puis les autres dans la boucle
            //puis la premiere a part
            temp = layers.back()->backward(layers[layers.size() - 2]->get_output(), temp);
            for (unsigned i = layers.size() - 2; i > 0; i--)
                temp = layers[i]->backward(layers[i - 1]->get_output(), temp);
        }
        layers[0]->backward(inputs, temp);
        //update des parametres
        update_params();
    }

#define EXTRACT_BATCH_FROM_RANK3_LAMBDA [](Tensor<Real>& inputs, const std::size_t batch_idx) {                        \
                            return inputs(batch_idx).as_matrix(inputs.get_dimensions()[1], inputs.get_dimensions()[2]);\
                            }

#define EXTRACT_BATCH_FROM_RANK5_LAMBDA [](Tensor<Real>& inputs, const std::size_t batch_idx) {                         \
                            return inputs(batch_idx).as_matrix(inputs.get_dimensions()[1],                              \
                                   inputs.get_dimensions()[2] * inputs.get_dimensions()[3] * inputs.get_dimensions()[4]);\
                            }

    template<typename Loss, typename T>
    inline void NeuralNetwork::train(Tensor<T>& inputs, Tensor<T>& labels, unsigned epochs) {
        assert(inputs.get_dimensions().size() == 3 || inputs.get_dimensions().size() == 5);

        unsigned i = 0;
        auto get_input_batch = inputs.get_dimensions().size() == 3 ?
                               EXTRACT_BATCH_FROM_RANK3_LAMBDA : EXTRACT_BATCH_FROM_RANK5_LAMBDA;

        for (unsigned ep = 0; ep < epochs; ++ep)
            for (std::size_t batch_idx = 0; batch_idx < inputs.get_dimensions()[0]; ++batch_idx) {
                //on extrait le batch du tenseur inputs qu'on stocke dans une matrice
                const auto batch = get_input_batch(inputs, batch_idx);
                const auto labels_batch = labels(batch_idx).as_matrix();

                //on effectue le feedforward
                auto output = layers[0]->forward(batch);
                for (unsigned i = 1; i < layers.size(); i++) {
                    /*Real _mean = mean(output);
                    Real _stddev = stddev(output, _mean);
                    std::cout << "Mean : " << _mean << ", std : " << _stddev << std::endl;*/
                    output = layers[i]->forward(output);
                }
                /*Real _mean = mean(output);
                Real _stddev = stddev(output, _mean);
                std::cout << "Mean : " << _mean << ", std : " << _stddev << std::endl;*/

                //on calcule l'erreur
                //Real loss = (output-labels_batch).array().square().sum();
                Real loss = Loss::forward(output, labels_batch);

                backpropagation<Loss>(batch, output, labels_batch);
                std::cout << "Iteration : " << i << ", loss(total)=" << loss
                          << std::endl;
                ++i;

            }
    }

    template<typename Loss, typename T>
    inline void NeuralNetwork::train(Tensor<T>& inputs, Tensor<T>& labels, unsigned epochs,
                                     std::function<Real(const Real, const unsigned)>& lr_updater) {
        assert(inputs.get_dimensions().size() == 3 || inputs.get_dimensions().size() == 5);

        unsigned i = 0;
        auto get_input_batch = inputs.get_dimensions().size() == 3 ?
                               EXTRACT_BATCH_FROM_RANK3_LAMBDA : EXTRACT_BATCH_FROM_RANK5_LAMBDA;

        for (unsigned ep = 0; ep < epochs; ++ep)
            for (std::size_t batch_idx = 0; batch_idx < inputs.get_dimensions()[0]; ++batch_idx) {
                //on extrait le batch du tenseur inputs qu'on stocke dans une matrice
                const auto batch = get_input_batch(inputs, batch_idx);
                const auto labels_batch = labels(batch_idx).as_matrix();

                //on effectue le feedforward
                auto output = layers[0]->forward(batch);
                for (unsigned i = 1; i < layers.size(); i++) {
                    output = layers[i]->forward(output);
                }

                //Calcul de l'erreur
                Real loss = Loss::forward(output, labels_batch);

                backpropagation(batch, output, labels_batch);
                std::cout << "Iteration : " << i << ", loss(total)=" << loss
                          << std::endl;

                // Mise a jour du learning rate
                for (auto& layer : layers)
                    layer->set_lr(lr_updater(layer->get_lr(), i));
                ++i;
            }
    }

    template<typename Loss>
    inline void NeuralNetwork::train(Dataset& dataset, const unsigned epochs) {
        train<Loss>(dataset.training_data, dataset.training_labels, epochs);
    }

    template<typename Loss>
    inline void NeuralNetwork::train(Dataset& dataset, const unsigned epochs,
                                     std::function<Real(const Real, const unsigned)> lr_updater) {
        train<Loss>(dataset.training_data, dataset.training_labels, epochs, lr_updater);
    }

    void NeuralNetwork::save(const std::string& path) const {
        std::ofstream file;
        file.open(path, std::ios::trunc);
        if (file.is_open()) {
            for (const auto& layer : layers) {
                file << layer->get_name() << '\n';
            }
            for (const auto& layer : layers) {
                layer->print_parameters(file);
            }
        }
    }

    inline void NeuralNetwork::load(const std::string& path) {
        std::ifstream file;
        file.open(path);
        if (file.is_open()) {
            std::string line;
            for (unsigned i = 0; i < layers.size(); ++i) {
                line = "";
                std::getline(file, line);
                layers[i]->load_hyper_parameters(line);
                layers[i]->check_hyper_parameters(line);
            }
            //Charger parametres
            for (const auto& layer : layers) {
                layer->load_parameters(file);
            }
        }
    }

} // namespace nnl

#endif //NNL_NEURALNETWORK_HPP
