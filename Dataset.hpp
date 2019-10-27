
#ifndef NNL_DATASET_HPP
#define NNL_DATASET_HPP

#include "Tensor.hpp"
#include "Transform.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace nnl {

/*
 * La classe dataset permet de stocker en un seul objet les donnees de training
 * et de test, qu'il s'agisse d'entrees ou d'etiquettes.
 * Elle permet aussi d'appliquer des transformations aux donnees.
 */
    struct Dataset {
        Tensor<Real> training_labels;
        Tensor<Real> training_data;
        Tensor<Real> test_labels;
        Tensor<Real> test_data;

        std::vector<std::unique_ptr<Transform> > data_tfm;
        std::vector<std::unique_ptr<Transform> > labels_tfm;


        template<typename T>
        inline void add_data_transforms(T&& tfm);

        /*-*
         * Permet d'ajouter des transformations pour les donnees qui seront effectuees lors d'un appel a
         * <code>apply_data_transforms</code>.
         *-*/
        template<typename TFM, typename ...TFMPack>
        inline void add_data_transforms(TFM&& tfm, TFMPack&& ...tfms);

        template<typename T>
        inline void add_labels_transforms(T&& tfm);

        /*-*
         * Permet d'ajouter des transformations pour les etiquettes qui seront effectuees lors d'un appel a
         * <code>apply_label_transforms</code>.
         *-*/
        template<typename TFM, typename ...TFMPack>
        inline void add_labels_transforms(TFM&& tfm, TFMPack&& ...tfms);

        /*-*
         * Applique aux donnees les transformations stockees dans <code>data_tfm</code>. Les arguments <code>training</code>
         * et <code>test</code>
         * permettent de choisir d'appliquer les transformations uniquement aux donnees d'entrainement ou aussi a celles
         * de test.
         *-*/
        inline void apply_data_transforms(const bool training = true, const bool test = true);

        /*-*
         * Applique aux etiquettes les transformations stockees dans <code>labels_tfm</code>. Les arguments <code>training</code>
         * et <code>test</code>
         * permettent de choisir d'appliquer les transformations uniquement aux etiquettes d'entrainement ou aussi a celles
         * de test.
         *-*/
        inline void apply_labels_transforms(const bool training = true, const bool test = true);
    };


    template<typename T>
    inline void Dataset::add_data_transforms(T&& tfm) {
        data_tfm.push_back(std::make_unique<T>(tfm));
    }

    template<typename TFM, typename ...TFMPack>
    inline void Dataset::add_data_transforms(TFM&& tfm, TFMPack&& ...tfms) {
        data_tfm.push_back(std::make_unique<TFM>(tfm));
        add_data_transforms(std::forward<TFMPack>(tfms)...);
    }


    template<typename T>
    inline void Dataset::add_labels_transforms(T&& tfm) {
        labels_tfm.push_back(std::make_unique<T>(tfm));
    }

    template<typename TFM, typename ...TFMPack>
    inline void Dataset::add_labels_transforms(TFM&& tfm, TFMPack&& ...tfms) {
        labels_tfm.push_back(std::make_unique<TFM>(tfm));
        add_labels_transforms(std::forward<TFMPack>(tfms)...);
    }

    inline void Dataset::apply_data_transforms(const bool training, const bool test) {
        if (training)
            for (const auto& tfm : data_tfm)
                tfm->encode(training_data);

        if (test)
            for (const auto& tfm : data_tfm)
                tfm->encode(test_data);
    }

    inline void Dataset::apply_labels_transforms(const bool training, const bool test) {
        if (training)
            for (const auto& tfm : labels_tfm)
                tfm->encode(training_labels);

        if (test)
            for (const auto& tfm : labels_tfm)
                tfm->encode(test_labels);
    }

} //namespace nnl

#endif //NNL_DATASET_HPP
