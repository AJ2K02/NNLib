#ifndef NNL_FULLYCONNECTED_HPP
#define NNL_FULLYCONNECTED_HPP

#include "Activations.hpp"
#include "config.h"
#include "Layer.hpp"
#include "utils.hpp"

#include <cmath>
#include <string>
#include <utility>


namespace nnl {

/*
* A COMMENTER
*/
    template<typename Activation>
    class FullyConnected : public Layer {
    public:
        //-* Constructeur par defaut.
        FullyConnected() = default;

        /*-*
         * Constructeur prenant deux parametres :<br>
         * <code>inputs</code> : Nombre de valeurs d'entree ;<br>
         * <code>outputs</code> : Nombre de neurones de la couche.
         *-*/
        FullyConnected(const unsigned inputs, const unsigned outputs);

        //-* Constructeur par copie.
        FullyConnected(const FullyConnected& rhs) = default;

        //-* Operateur d'assignement par copie.
        FullyConnected& operator=(const FullyConnected& rhs);

        //-* Retourne le nom de la couche (utilise pour sauvegarder le reseau de neurone)
        std::string get_name() const override {
            return "FullyConnected{" + std::to_string(weights.rows()) +
                   ',' + std::to_string(weights.cols()) + ',' +
                   std::to_string(lr) + ',' + Activation::get_name() + '}';
        };

        //-* Retourne la matrice de poids.
        Matrix get_weights() const override { return weights; }

        //-* Defini la matrice de poids.
        void set_weights(const Matrix& new_weights) { weights = new_weights; }

        //-* Retourne la matrice de gradient de l'erreur par rapport aux poids.
        Matrix get_delta_w() const override { return delta_w; }

        //-* Retourne le vecteur d'unites de biais.
        EVector get_biases() const override { return biases; }

        //-* Defini le vecteur d'unites de biais.
        void set_biases(const EVector& new_biases) { biases = new_biases; }

        //-* Retourne la matrice de gradient de l'erreur par rapport aux unites de biais.
        EVector get_delta_b() const override { return delta_b; }

        //-* Retourne le taux d'apprentissage de la couche.
        Real get_lr() const override { return lr; }

        //-* Permet de definir le taux d'apprentissage.
        void set_lr(const Real new_lr) override { lr = new_lr; }

        //-* Retourne la matrice de pre-activation resultant de la derniere utilisation de la couche.
        Matrix get_pre_act() const override { return z; }

        //-* Retourne la matrice de sortie resultant de la derniere utilisaion de la couche.
        Matrix get_output() const override { return a; }

        //-* Retourne le nombre de valeurs attendues en entree de la couche.
        unsigned get_inputs_nb() const override { return weights.rows(); }

        //-* Retourne le nombre de valeurs attendues en entree de la couche.
        unsigned get_outputs_nb() const override { return weights.cols(); }


        /*-*
         * Permet de geler ou de de-geler le calcul des gradients au sein de la couche.
         *-*/
        void freeze(const bool v = true) { m_frozen = v; }

        /*-*
         * Calcule l'output de la couche pour une matrice d'entree <code>inputs</code>.
         *-*/
        Matrix forward(const Matrix& inputs);

        /*-*
         * Calcule les gradients de l'erreur par rapport aux poids et aux biais.
         * <code>inputs</code> est la matrice d'inputs qui avait ete passee a la couche ;<br>
         * <code>dL_over_dy</code> est la matrice des derivees de L (l'erreur) par rapport a la sortie de la couche.
         *-*/
        Matrix backward(const Matrix& inputs, const Matrix& dL_over_dy);

        /*-*
         * Met a jour les poids et les biais de la couche grace aux gradients, qui doivent avoir etes calcules au prealable.
         *-*/
        void update_param();

        //-* Ecrit les parametres de la couche dans un flux
        void print_parameters(std::ostream& s) const;

        //-* Charge les parametres a partir d'un flux
        void load_parameters(std::ifstream& file);

        //-* Verifie que les hyper-parametres correspondent a la ligne chargee depuis le fichier
        void check_hyper_parameters(const std::string& line) const;

        //-* Charge les hyper-parametres necessaires depuis la ligne du fichier, ici le learning rate.
        void load_hyper_parameters(const std::string& line);

    private:
        Matrix weights;
        EVector biases;

        Real lr = 1.0;

        //cache pour garder les valeurs avant la retropropagation:
        //pre activation
        Matrix z;
        //activation
        Matrix a;

        //valeurs utilisees pour updater weights et biases lors de la retro-propagation
        Matrix delta_w;
        EVector delta_b;

        bool m_frozen = false;
    };


    template<typename Activation>
    FullyConnected<Activation>::FullyConnected(const unsigned inputs, const unsigned outputs) {
        //Xavier initialization
        //TODO laisser choix du type initialisation a l'utilisateur...
        weights = decltype(weights)(inputs, outputs);
        auto randn_ = std::bind(randn, 0, std::sqrt(2.0 / (inputs + outputs)));
        for (unsigned i = 0; i < inputs * outputs; ++i)
            weights.data()[i] = randn_();
        biases = EVector::Zero(outputs);
    }

/*
template<typename Activation>
void FullyConnected<Activation>::init(const unsigned inputs, const unsigned outputs) {
    weights = Matrix::Random(inputs, outputs);
    biases = EVector::Zero(outputs);
}*/


    template<typename Activation>
    FullyConnected<Activation>& FullyConnected<Activation>::operator=(const FullyConnected& rhs) {
        weights = rhs.get_weights();
        biases = rhs.get_biases();
        lr = rhs.get_lr();
        z = rhs.get_pre_act();
        a = rhs.get_output();
        delta_w = rhs.get_delta_w();
        delta_b = rhs.get_delta_b();
        return *this;
    }


    template<typename Activation>
    Matrix FullyConnected<Activation>::forward(const Matrix& inputs) {
        //.rowwise() ==> on ajoute le vecteur biases a chaque ligne
        z = (inputs * weights).rowwise() + biases.transpose();
        //on applique la fonction d'activation
        a = z.unaryExpr(std::ref(Activation::forward));
        return a;
    }


//retro-propagation, on calcul les matrices delta_w et delta_b
//inputs : inputs qui avaient ete passe a la couche lors du feed-forward
//dL_over_dy : derivee de l'erreur finale par rapport à
//l'activation de la couche (ici "a")
//Return : dL/dy pour la couche precedente dans le reseau de neurone
    template<typename Activation>
    Matrix FullyConnected<Activation>::backward(const Matrix& inputs, const Matrix& dL_over_dy) {
        Matrix dL_over_dz = dL_over_dy.cwiseProduct(z.unaryExpr(std::ref(Activation::backward)));
        if (!m_frozen) {
            delta_w = inputs.transpose() * dL_over_dz;
            delta_b = dL_over_dz.colwise().sum().transpose();
        }
        return dL_over_dz * weights.transpose();
    }


    template<typename Activation>
    void FullyConnected<Activation>::update_param() {
        if (!m_frozen) {
            weights -= lr * delta_w;
            biases -= lr * delta_b;
        }
    }


    template<typename Activation>
    void FullyConnected<Activation>::print_parameters(std::ostream& s) const {
        for (std::size_t i = 0; i < weights.cols() * weights.rows(); ++i) {
            s << weights.data()[i] << ' ';
        }
        for (std::size_t i = 0; i < biases.rows(); ++i) {
            s << biases[i] << ' ';
        }
        s << std::endl; // '\n' + flush
    }


    template<typename Activation>
    void FullyConnected<Activation>::load_parameters(std::ifstream& file) {
        for (std::size_t i = 0; i < weights.rows() * weights.cols(); ++i) {
            file >> weights.data()[i];
            file.ignore(); //on ignore l'espace
        }
        for (std::size_t i = 0; i < biases.rows(); ++i) {
            file >> biases[i];
            file.ignore();
        }
        file.ignore(); // fin de ligne
    }


    template<typename Activation>
    void FullyConnected<Activation>::check_hyper_parameters(const std::string& line) const {
        // stringstream contenant la ligne depuis le premier hyper parametre
        std::stringstream ss(std::string(std::find(line.begin(), line.end(), '{') + 1, line.end()));

        unsigned in, out;
        double eta;
        ss >> in;
        ss.ignore();
        ss >> out;
        ss.ignore();
        ss >> eta;
        assert(in == weights.rows() && out == weights.cols() && "Layer sizes don't match");
        assert(eta == lr && "Learning rate doesn't match");
    }


    template<typename Activation>
    void FullyConnected<Activation>::load_hyper_parameters(const std::string& line) {
        // stringstream contenant la ligne a partir du premier hyper parametre
        std::stringstream ss(std::string(std::find(line.begin(), line.end(), '{') + 1, line.end()));
        int dump;
        ss >> dump;
        ss.ignore();
        ss >> dump;
        ss.ignore();
        // Seul le learning rate nous interesse
        ss >> lr;
    }

} // namespace nnl

#endif //NNL_FULLYCONNECTED_HPP
