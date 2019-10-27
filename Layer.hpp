#ifndef NNL_LAYER_HPP
#define NNL_LAYER_HPP

#include "config.h"
#include "Eigen/Dense"
#include "Tensor.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace nnl {


/*
* class Layer :
*
* classe de base pour toute les couches
*	FullyConnected, ConvLayer, etc heritent
*	de Layer
*
* A faire (peut etre)?
* 1:
* 	Plutot que d'utiliser des fonctions virtuelles,
* 	utiliser le polymorphisme statique avec les
* 	templates.
*   23.07.19 : En fait pas tres utile..?
*
* 2:
*		Avoir une classe Tensor au lieu de Matrix,
*   EVector, etc. Permettrait d'avoir le meme type
*   de retour des fonctions pour des couches
*   du style FullyConnected ou Convolutional
*
* Pour creer un nouveau type de couche :
* la faire heriter (public) de Layer,
* et overrider toutes les fonctions virtual
*
*/
    class Layer {
    public:
        //virtual void init(const unsigned inputs, const unsigned outputs) { std::cerr << "Oops, function not overridden\n"; }


        //getters / setters

        //-* Retourne le nom de la couche (pour sauvergarder la couche dans un fichier texte)
        virtual std::string get_name() const { std::cerr << "Oops, function get_name not overriden\n"; };

        //-* Retourne la matrice de poids.
        virtual Matrix get_weights() const { std::cerr << "Oops, function get_weights not overridden\n"; }

        //-* Retourne la matrice du gradient de l'erreur par rapport aux poids.
        virtual Matrix get_delta_w() const { std::cerr << "Oops, function get_delta_w not overridden\n"; }

        //-* Retourne le vecteur d'unités de biais.
        virtual EVector get_biases() const { std::cerr << "Oops, function get_biases not overridden\n"; }

        //-* Retourne le vecteur du gradient de l'erreur par rapport aux unités de biais.
        virtual EVector get_delta_b() const { std::cerr << "Oops, function get_delta_b not overridden\n"; }

        //-* Retourne le taux d'apprentissage.
        virtual Real get_lr() const { std::cerr << "Oops, function get_lr not overridden\n"; }

        //-* Défini le taux d'apprentissage.
        virtual void set_lr(const Real new_lr) { std::cerr << "Oops, function set_lr not overridden\n"; }

        //-* Retourne la matrice de préactivation de la dernière utilisation de la couche.
        virtual Matrix get_pre_act() const { std::cerr << "Oops, function get_pre_act not overridden\n"; }

        //-* Retourne la matrice de sortie de la dernière utilisation de la couche.
        virtual Matrix get_output() const { std::cerr << "Oops, function get_output not overridden\n"; }

        //-* Retourne le nombre de valeurs attendues en entrée.
        virtual unsigned get_inputs_nb() const { std::cerr << "Oops, function get_inputs_nb not overriden\n"; }

        //-* Retourne le nombre de valeurs de sortie.
        virtual unsigned get_outputs_nb() const { std::cerr << "Oops, function get_outputs_nb not overriden\n"; }

        //-* Permet de geler ou de dégeler la couche selon que %%v vaille 0 ou 1 (gelé = les paramètres ne seront pas mis à jour lors de la rétro-propagation.)
        virtual void freeze(const bool v) { std::cerr << "Oops, function freeze not overridden\n"; }

        //feed forward et retropropagation

        //-* Méthode effectuant la propagation-avant avec %%inputs comme matrice d'entrée.
        virtual Matrix forward(const Matrix& inputs) { std::cerr << "Oops, function forward not overridden\n"; }

        /*-*
         * Méthode effectuant la rétro-propagation avec %%inputs comme matrice d'entrée et %%dL_over_dy étant la
         * matrice contenant le gradient de l'erreur par rapport aux sorties de la couche.
         *-*/
        virtual Matrix backward(const Matrix& inputs, const Matrix& dL_over_dy) {
            std::cerr << "Oops, function backward not overridden\n";
        }

        //-* Met à jour les paramètres en utilisant les gradients, calculés au préalable.
        virtual void update_param() { std::cerr << "Oops, function update_param not overridden\n"; }

        //-* Ecrit les paramètres de la couche dans le flux &&s.
        virtual void print_parameters(std::ostream& s) const {
            std::cerr << "Oops, function print_parameters not overridden\n";
        }

        //-* Charge les paramètres depuis le flux %%file.
        virtual void load_parameters(std::ifstream& file) {
            std::cerr << "Oops, function load_parameters not overridden\n";
        }

        /*-*
         * Vérifie que les hyper-paramètres de la couche correspondent à ceux écrits dans la chaîne de caractères %%line.
         * Note : les vérifications sont effectuées dans des assertions (%%assert(...)).
        *-*/
        virtual void check_hyper_parameters(const std::string& line) const {
            std::cerr << "Oops, function load_parameters not overridden\n";
        }

        //-* Charge les hyper_paramètres à partir d'une chaîne de caractère %%line.
        virtual void load_hyper_parameters(const std::string& line) {
            std::cerr << "Oops, function load_parameters not overridden\n";
        }
    };

} //namespace

#endif //NNL_LAYER_HPP