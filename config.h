
/*
 *  Ce fichier contient les differents aliases (using xxx = yyy) utilises globalement dans la
 *  librairie. Il est donc inclu dans a peu pres tous les fichiers.
 *  Il peut aussi contenir des define utilise globalement.
 */

#ifndef NNL_CONFIG_H
#define NNL_CONFIG_H

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace nnl {

    using Real = double;
#define _REAL_MIN std::numeric_limits<Real>::min()
#define _REAL_MAX std::numeric_limits<Real>::max()

/*
 * Note : Matrix et RMMat sont identiques... Pour plus de coherence :
 * TODO peut-etre remplacer tout les Matrix par des RMMat ...?
 */
    using Matrix = Eigen::Matrix<Real,            //Matrice (dense) de taille dynamique stockee ligne par ligne
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor>;

    template<typename Scalar, Eigen::StorageOptions StorageOpt>
    using Mat = Eigen::Matrix<Scalar,            //Matrice (dense) de taille dynamique
            Eigen::Dynamic,
            Eigen::Dynamic,
            StorageOpt>;

    using Array = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>;

//Matrice Row-Major
    template<typename Scalar>
    using RMMat = Mat<Scalar, Eigen::RowMajor>; //Matrice (dense) de taille dynamique stockee ligne par ligne

    using EVector = Eigen::VectorXd;

    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>; //utilise pour les couches conv

    using Triplet = Eigen::Triplet<double>;                     // ^

//TODO : definir un NOMLIB_ASSERT(x) si un NOMLIB_DEBUG a ete defini : permet de desactiver les assert en release
//ou a l'utilisateur de choisir de les activer ou non.

    enum ImageDataStorage {
        NonLinearizedImages,
        LinearizedImages
    };

} // namespace nnl

#endif //NNL_CONFIG_H
