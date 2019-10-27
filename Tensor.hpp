#ifndef NNL_TENSOR_HPP
#define NNL_TENSOR_HPP

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "utils.hpp"

namespace nnl {

/*
 * En c++17 il existe deja un std::conjunction, mais refait
 * pour la compatibilite avec des versions anterieures.
 * Vaut true seulement si tout les parametres du pack
 * ont un membre statique value valant true.
 * conjunction sera utilise avec des std::is_convertible,
 * qui ont bien un membre statique de type bool.
 */
    template<bool...>
    struct bool_only_pack {
    };
    template<typename ...T_pack>
    using conjunction = typename std::is_same<
            bool_only_pack<true, T_pack::value...>,
            bool_only_pack<T_pack::value..., true> >;


//Permet de "desactiver" une fonction si les types passes dans le pack
//d'argument ne sont pas tous convertibles en RequiredType
    template<typename RequiredType, typename ...T_pack>
    using all_scalar_types = typename std::enable_if<
            conjunction<std::is_convertible<
                    T_pack, RequiredType>...>::value>::type;


/*
* Classe Tensor :
*   Type utilise lorsque l'utilisateur veut passer des donnees au
*   reaseau de neurone : par exemple les inputs seront un Tensor, les
*   outputs aussi, les noyaux dans les couches convolutives, etc.
*
*   Contient les donnees dans un tableau style C, et un std::vector avec
*   la taille dans chaque dimension.
*
*/
    template<typename Scalar>
    class Tensor {
    public:

        inline explicit Tensor() : m_data(nullptr) {}

        inline explicit Tensor(const std::vector<std::size_t>& dims);

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...>>
        inline explicit Tensor(const Scalar_pack ...dims);

        inline explicit Tensor(Scalar* data, const std::vector<std::size_t>& dims) : m_data(data), m_dims(dims),
                                                                                     m_owns_rc(false) {}

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...>>
        inline explicit Tensor(const RMMat <Scalar>& mat,
                               const Scalar_pack ...dims);

        inline explicit Tensor(const RMMat <Scalar>& mat);

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...>>
        inline explicit Tensor(const Mat <Scalar, Eigen::ColMajor>& mat,
                               const Scalar_pack ...dims);

        inline explicit Tensor(const std::vector<std::size_t>& dims, const Scalar val);

        inline Tensor(const Tensor<Scalar>& tensor);

        inline ~Tensor();

        inline Tensor<Scalar>& operator=(const Tensor& rhs);

        inline void copy_data(const Tensor<Scalar>& rhs);

        inline Tensor<Scalar>& operator=(const std::initializer_list<Scalar>& list);

        inline Tensor<Scalar>& operator+=(const Tensor<Scalar>& rhs);

        //-* Additionne le nombre %%rhs à chaque coefficient.
        inline Tensor<Scalar>& operator+=(const Scalar rhs);

        inline Tensor<Scalar>& operator-=(const Tensor<Scalar>& rhs);

        //-* Soustrait le nombre %%rhs à chaque coefficient.
        inline Tensor<Scalar>& operator-=(const Scalar rhs);

        //-* Retourne la somme coefficient à coefficient du %%Tensor et de %%rhs.
        inline Tensor<Scalar> operator+(const Tensor<Scalar>& rhs) const;

        //-* Retourne le %%Tensor, dont on a additionné %%rhs à chaque coefficient.
        inline Tensor<Scalar> operator+(const Scalar rhs) const;

        //-* Retourne le %%Tensor, dont on a soustrait %%rhs à chaque coefficient.
        inline Tensor<Scalar> operator-(const Scalar rhs) const;

        //-* Multiplie chaque coefficient par le coefficient correspondant dans le %%Tensor %%rhs.
        inline Tensor<Scalar>& operator*=(const Tensor<Scalar>& rhs);

        //-* Divise chaque coefficient par le coefficient correspondant dans le %%Tensor %%rhs.
        inline Tensor<Scalar>& operator/=(const Tensor<Scalar>& rhs);

        //-* Retourne le produit coefficient à coefficient du %%Tensor avec le %%Tensor %%rhs.
        inline Tensor<Scalar> operator*(const Tensor<Scalar>& rhs) const;

        //-* Retourne le quotient coefficient à coefficient du %%Tensor avec le %%Tensor %%rhs.
        inline Tensor<Scalar> operator/(const Tensor<Scalar>& rhs) const;

        //-* Multiplie chaque coefficient par le scalaire %%rhs.
        inline Tensor<Scalar>& operator*=(const Scalar rhs);

        //-* Divise chaque coefficient par le scalaire %%rhs.
        inline Tensor<Scalar>& operator/=(const Scalar rhs);

        //-* Retourne le %%Tensor, dont chaque coefficient a été multiplié par le scalare %%rhs.
        inline Tensor<Scalar> operator*(const Scalar rhs) const;

        //-* Retourne le %%Tensor, dont chaque coefficient a été divisé par le scalare %%rhs.
        inline Tensor<Scalar> operator/(const Scalar rhs) const;

        inline Scalar& operator[](const std::vector<std::size_t>& idx);

        template<typename ...IdxPack,
                typename = all_scalar_types<std::size_t, IdxPack...>>
        inline Tensor<Scalar> operator()(IdxPack ...idx);

        template<typename ...IdxPack,
                typename = all_scalar_types<std::size_t, IdxPack...>>
        inline const Tensor<Scalar> operator()(IdxPack ...idx) const;

        inline Tensor<Scalar>& plus_n_times(const Tensor<Scalar>& rhs, const Scalar N);

        inline Tensor<Scalar>& minus_n_times(const Tensor<Scalar>& rhs, const Scalar N);

        /*-*
         * Appelle le functor %%to_call pour chaque coefficient.
         * %%to_call doit prendre un %%Scalar& en paramètre et ne retourne rien
         * (ou sa valeur de retour sera ignorée).
        *-*/
        template<typename Callable>
        inline Tensor<Scalar>& foreach(const Callable& to_call);

        //-* Met chaque coefficient du %%Tensor au carré.
        inline Tensor<Scalar>& square();

        //-* Remplace chaque coefficient par sa racine carrée.
        inline Tensor<Scalar>& sqrt();

        //-* Met chaque coefficient du %%Tensor à la puissance %%x.
        inline Tensor<Scalar>& power(const Scalar x);

        //-* Retourne la moyenne ou le centre des coefficients du %%Tensor.
        inline Scalar mean() const;

        //-* Retourne la variance des coefficients du %%Tensor.
        inline Scalar std() const;

        //-* Retourne l'écart-type des coefficients du %%Tensor autour de la moyenne passée dans l'argument %%_mean.
        inline Scalar std(const Scalar _mean) const;

        //-* Retourne la variance des coefficients du %%Tensor.
        inline Scalar var() const;

        //-* Retourne la variance des coefficients du %%Tensor autour de la moyenne passée dans l'argument %%_mean.
        inline Scalar var(const Scalar _mean) const;

        //-* Retourne la somme des coefficients du %%Tensor.
        inline Scalar sum();

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...> >
        inline Tensor<Scalar>& resize(const Scalar_pack ...dims);

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...> >
        inline Tensor<Scalar> reshape(const Scalar_pack ...dims) const;

        /*-*
         * Retourne une copie du Tensor redimensionnée. L'objet à partir duquel
         * la méthode est appelée n'est pas modifié. Cette méthode doit être
         * utilisée si le nombre d'éléments total ne change pas.
         * Cette méthode prend en paramètre un %%vector contenant les nouvelles dimensions.
        *-*/
        inline Tensor<Scalar> reshape(const std::vector<std::size_t>& dims) const;

        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...> >
        inline Tensor<Scalar>& reshape_in_place(const Scalar_pack ...dims);

        /*-*
         * Retourne une copie du Tensor redimensionnée. L'objet à partir
         * duquel la méthode est lui aussi redimensionné. Cette méthode doit
         * être utilisée si le nombre d'éléments total ne change pas. Sinon,
         * voir resize. Cette méthode prend en paramètre un %%vector contenant
         * les nouvelles dimensions.
        *-*/
        inline Tensor<Scalar>& reshape_in_place(const std::vector<std::size_t>& dims);

        //-* Méthode identique à %%reshape_in_place, mais ne retourne rien.
        template<typename ...Scalar_pack,
                typename = all_scalar_types<std::size_t, Scalar_pack...> >
        inline void reshape_in_place_no_return(const Scalar_pack ...dims);

        //-* Méthode identique à %%reshape_in_place, mais ne retourne rien.
        inline void reshape_in_place_no_return(const std::vector<std::size_t>& dims);

        inline Eigen::Map<RMMat < Scalar> >

        as_matrix(const std::size_t rows,
                  const std::size_t cols);


        inline Eigen::Map<RMMat < Scalar> > as_matrix();

        //-* Assigne la valeur %%val aux %%size premiers éléments.
        inline Tensor<Scalar>& set_value(const Scalar val, const std::size_t size);

        //-* Assigne la valeur %%val à tous les éléments.
        inline Tensor<Scalar>& set_value(const Scalar val);

        //-* Assigne 0 à tous les éléments.
        inline Tensor<Scalar>& set_zero();

        //-* ASsigne 1 à tous les éléments.
        inline Tensor<Scalar>& set_one();

        inline const std::vector<std::size_t>& get_dimensions() const { return m_dims; }

        inline Scalar* data() const { return m_data; }

        inline Scalar& item() { return *m_data; }

        //TODO Doc
        inline const Scalar& item() const { return *m_data; }

        inline std::size_t linearize_index(const std::vector<std::size_t>& idx) const;

        //-* Retourne l'index linéarisé de l'élément le plus gran contenu.
        inline std::size_t argmax() const;

        inline void dprint();

        inline void print();

    private:
        std::vector<std::size_t> m_dims;
        Scalar* m_data = nullptr;
        bool m_owns_rc = true;
    };


    template<typename Scalar>
    inline Tensor<Scalar>::Tensor(const std::vector<std::size_t>& dims)
            : m_dims(dims) {
        unsigned total_elements = std::accumulate(m_dims.begin(),
                                                  m_dims.end(), 1,
                                                  std::multiplies<std::size_t>());
        m_data = new Scalar[total_elements];
    }


    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
//fonction marche que si tous les nombres sont convertibles en Scalar
    inline Tensor<Scalar>::Tensor(const Scalar_pack ...dims) {
        m_dims = {static_cast<std::size_t>(dims)...};
        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());
        m_data = new Scalar[total_elements];
    }


    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline Tensor<Scalar>::Tensor(const RMMat <Scalar>& mat,
                                  const Scalar_pack ...dims) {
        m_dims = {static_cast<std::size_t>(dims)...};
        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());
        assert(total_elements == mat.rows() * mat.cols() && "Sizes don't match");
        m_data = new Scalar[total_elements];

        std::copy(mat.data(), mat.data() + total_elements, m_data);
    }


    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline Tensor<Scalar>::Tensor(const Mat <Scalar, Eigen::ColMajor>& mat,
                                  const Scalar_pack ...dims) {
        m_dims = {static_cast<std::size_t>(dims)...};

        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());
        assert(total_elements == mat.rows() * mat.cols() && "Sizes don't match");

        assert(false && "NOT IMPLEMENTED YET, TODO");
    }


    template<typename Scalar>
    inline Tensor<Scalar>::Tensor(const RMMat <Scalar>& mat) {
        //mat.rows est de type Eigen::Index, soit normalement std::ptr_diff_t,
        //qui est un type signe. Cependant on part du principe que le nombre
        //de colonnes et de lignes est non-signe.
        m_dims.resize(2);
        m_dims = {static_cast<std::size_t>(mat.rows()), static_cast<std::size_t>(mat.cols())};

        std::size_t total_elements = mat.rows() * mat.cols();
        m_data = new Scalar[total_elements];

        std::copy(mat.data(), mat.data() + total_elements, m_data);
    }


    template<typename Scalar>
    inline Tensor<Scalar>::Tensor(const std::vector<std::size_t>& dims,
                                  const Scalar val)
            : m_dims(dims) {
        unsigned total_elements = std::accumulate(m_dims.begin(),
                                                  m_dims.end(),
                                                  1u,
                                                  std::multiplies<std::size_t>());
        m_data = new Scalar[total_elements];
        set_value(val, total_elements);
    }


//Constructeur par copie
//Copie les nombres dans le nouveau tenseur, pas le pointeur m_data
    template<typename Scalar>
    inline Tensor<Scalar>::Tensor(const Tensor<Scalar>& tensor) {
        m_dims = tensor.get_dimensions();
        unsigned total_elements = std::accumulate(m_dims.begin(),
                                                  m_dims.end(), 1,
                                                  std::multiplies<std::size_t>());
        m_data = new Scalar[total_elements];
        std::copy(tensor.data(), tensor.data() + total_elements, m_data);
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator=(const Tensor& rhs) {
        m_dims = rhs.get_dimensions();
        unsigned total_elements = std::accumulate(m_dims.begin(),
                                                  m_dims.end(), 1,
                                                  std::multiplies<std::size_t>());
        m_data = new Scalar[total_elements];
        m_owns_rc = true;
        const Scalar* rhs_data = rhs.data();
        std::copy(rhs_data, rhs_data + total_elements, m_data);
        return *this;
    }

/*
 * TODO DOC
 * Copie les donnes du Tensor rhs dans *this.
 * Ne fait pas d'allocation et ne change pas m_owns_rc.
 */
    template<typename Scalar>
    inline void Tensor<Scalar>::copy_data(const Tensor<Scalar>& rhs) {
        assert(m_dims == rhs.get_dimensions() && "Sizes must be the same.");
        m_dims = rhs.get_dimensions();
        unsigned total_elements = std::accumulate(m_dims.begin(),
                                                  m_dims.end(), 1,
                                                  std::multiplies<std::size_t>());
        std::copy(rhs.data(), rhs.data() + total_elements, m_data);
    }


    template<typename Scalar>
    inline Tensor<Scalar>&
    Tensor<Scalar>::operator=(const std::initializer_list<Scalar>& list) {
        assert(list.size() == std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<>()));
        std::size_t i = 0;
        //std::cout << list.size() << std::endl;
        for (auto it = list.begin(); it < list.end(); ++i, ++it)
            m_data[i] = *it;
        return *this;
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator+=(const Tensor<Scalar>& rhs) {
        //assert(rhs.get_dimensions() == m_dims);
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] += rhs.data()[i];

        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator+(const Tensor<Scalar>& rhs) const {
        Tensor<Scalar> ret(*this);
        return ret += rhs;
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator+=(const Scalar rhs) {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        std::for_each(m_data, m_data + nb, [=](double& x) { x += rhs; });
        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator+(const Scalar rhs) const {
        Tensor<Scalar> ret(*this);
        return ret += rhs;
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator-=(const Tensor<Scalar>& rhs) {
        //assert(rhs.get_dimensions() == m_dims);
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] -= rhs.data()[i];

        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator-=(const Scalar rhs) {
        return operator+=(-rhs);
    }

    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator-(const Scalar rhs) const {
        return operator+(-rhs);
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator*=(const Tensor<Scalar>& rhs) {
        //assert(rhs.get_dimensions() == m_dims);
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] *= rhs.data()[i];

        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator/=(const Tensor<Scalar>& rhs) {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] /= rhs.data()[i];

        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator*(const Tensor<Scalar>& rhs) const {
        return Tensor<Scalar>(*this) *= rhs;
    }

    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator/(const Tensor<Scalar>& rhs) const {
        return Tensor<Scalar>(*this) /= rhs;
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator*=(const Scalar rhs) {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] *= rhs;

        return *this;
    }

//TODO DOC
    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::operator/=(const Scalar rhs) {
        return operator*=(1.0 / rhs);
    }

//TODO DOC
    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator*(const Scalar rhs) const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1,
                                         std::multiplies<std::size_t>());
        Tensor<Scalar> result(m_dims);
        for (unsigned i = 0; i < nb; ++i)
            result.data()[i] = m_data[i] * rhs;

        return result;
    }

//TODO DOC
    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::operator/(const Scalar rhs) const {
        return operator*(1.0 / rhs);
    }


    template<typename Scalar>
    inline Tensor<Scalar> operator+(const Tensor<Scalar>& lhs,
                                    const Tensor<Scalar>& rhs) {
        Tensor<Scalar> res(2, 3);
        std::size_t nb = std::accumulate(res.get_dimensions().begin(),
                                         res.get_dimensions().end(), 1,
                                         std::multiplies<std::size_t>());
        for (std::size_t i = 0; i < nb; ++i)
            res.data()[i] = lhs.data()[i] + rhs.data()[i];
        return res;
    }


    template<typename Scalar>
    inline Tensor<Scalar> operator-(const Tensor<Scalar>& lhs,
                                    const Tensor<Scalar>& rhs) {
        Tensor<Scalar> res(lhs.get_dimensions());
        std::size_t nb = std::accumulate(res.get_dimensions().begin(),
                                         res.get_dimensions().end(), 1,
                                         std::multiplies<std::size_t>());
        for (std::size_t i = 0; i < nb; ++i)
            res.data()[i] = lhs.data()[i] - rhs.data()[i];
        return res;
    }


//Permet d'updater les poids d'un reseau de neurone en soustrayant
//le gradient (de type Tensor) multiplie par le learning rate
//en parcourant les donnees uniquement une fois,
//contrairement a :
//      tenseur -= grad * learning_rate;
    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::plus_n_times(const Tensor<Scalar>& rhs,
                                                        const Scalar N) {
        assert(rhs.get_dimensions() == m_dims);
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        for (unsigned i = 0; i < nb; ++i)
            m_data[i] += rhs.data()[i] * N;
        return *this;
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::minus_n_times(const Tensor<Scalar>& rhs,
                                                         const Scalar N) {
        return plus_n_times(rhs, -N);
    }

//TODO DOC
    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::square() {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        std::for_each(m_data, m_data + nb, [&](Scalar& x) { x *= x; });
        return *this;
    }

//TODO DOC
    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::sqrt() {
        return this->power(0.5);
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::power(const Scalar x) {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        std::for_each(m_data, m_data + nb, [=](Scalar& s) { s = pow(s, x); });
        return *this;
    }

//TODO DOC
    template<typename Scalar>
    inline Scalar Tensor<Scalar>::mean() const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        Scalar m = std::accumulate(m_data, m_data + nb, Scalar(0)) / (double) nb;
        return m;
    }

//TODO DOC
    template<typename Scalar>
    inline Scalar Tensor<Scalar>::std() const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        const Scalar m = mean();
        Scalar sum = std::accumulate(m_data, m_data + nb, Scalar(0), [m](Scalar& a, Scalar& x) { return SQR(x - m); });
        return pow(sum / double(nb), 0.5);
    }

//TODO DOC
    template<typename Scalar>
    inline Scalar Tensor<Scalar>::std(const Scalar _mean) const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        Scalar sum = std::accumulate(m_data, m_data + nb, Scalar(0),
                                     [=](Scalar& a, Scalar& x) { return SQR(x - _mean); });
        return pow(sum / double(nb), 0.5);
    }

    template<typename Scalar>
    inline Scalar Tensor<Scalar>::var() const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        const Scalar m = mean();
        Scalar sum = std::accumulate(m_data, m_data + nb, Scalar(0), [m](Scalar& a, Scalar& x) { return SQR(x - m); });
        return sum / double(nb);
    }

    template<typename Scalar>
    inline Scalar Tensor<Scalar>::var(const Scalar _mean) const {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        Scalar sum = std::accumulate(m_data, m_data + nb, Scalar(0),
                                     [=](Scalar& a, Scalar& x) { return SQR(x - _mean); });
        return sum / double(nb);
    }

    template<typename Scalar>
    inline Scalar Tensor<Scalar>::sum() {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        return std::accumulate(m_data, m_data + nb, 0);
    }

//TODO doc
    template<typename Scalar>
    template<typename Callable>
    inline Tensor<Scalar>& Tensor<Scalar>::foreach(const Callable& to_call) {
        std::size_t nb = std::accumulate(m_dims.begin(),
                                         m_dims.end(), 1u,
                                         std::multiplies<std::size_t>());
        std::for_each(m_data, m_data + nb, to_call);
        return *this;
    }


/*
* Resize le tenseur, en changeant le nombre d'elements.
* Note : Les elements deja initialises seront perdus!!
*/
    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline Tensor<Scalar>& Tensor<Scalar>::resize(const Scalar_pack ...dims) {
        m_dims.resize(sizeof...(dims));
        m_dims = {static_cast<std::size_t>(dims)...};
        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(), 1u,
                                                     std::multiplies<std::size_t>());

        if (m_data != nullptr && m_owns_rc)
            delete[] m_data;                        // de-allouer le tableau actuel s'il a ete alloue
        m_data = new Scalar[total_elements];    // re-allouer un nouveau tableau
        return *this;
    }


//Fait une copie du tenseur et le reshape
//Impossible de retourner une reference a reshaped car cet objet est local,
//mais le compilateur devrait pouvoir eviter la copie en plus.
    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline Tensor<Scalar> Tensor<Scalar>::reshape(const Scalar_pack ...dims) const {
        Tensor<Scalar> reshaped(*this);
        reshaped.reshape_in_place(dims...);
        return reshaped;
    }

/*
 * TODO DOC
 */
    template<typename Scalar>
    inline Tensor<Scalar> Tensor<Scalar>::reshape(const std::vector<std::size_t>& dims) const {
        Tensor<Scalar> reshaped(*this);
        reshaped.reshape_in_place(dims);
        return reshaped;
    }

//Reshape l'objet lui-meme et le retourne par ref.
    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline Tensor<Scalar>& Tensor<Scalar>::reshape_in_place(const Scalar_pack ...dims) {
        m_dims = {static_cast<std::size_t>(dims)...};
        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::reshape_in_place(const std::vector<std::size_t>& dims) {
        m_dims = dims;
        return *this;
    }

//TODO DOC
//Reshape l'objet lui-meme et ne le retourne pas.
    template<typename Scalar>
    template<typename ...Scalar_pack, typename>
    inline void Tensor<Scalar>::reshape_in_place_no_return(const Scalar_pack ...dims) {
        m_dims = {static_cast<std::size_t>(dims)...};
    }

    template<typename Scalar>
    inline void Tensor<Scalar>::reshape_in_place_no_return(const std::vector<std::size_t>& dims) {
        m_dims = dims;
    }

    template<typename Scalar>
    inline Tensor<Scalar>::~Tensor() {
        if (m_data != nullptr && m_owns_rc)
            delete[] m_data;
    }


//Assigne une meme valeur a tous les elements du tenseur
//TODO : memset?
    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::set_value(const Scalar val,
                                                     const std::size_t size) {
        for (unsigned i = 0; i < size; i++)
            m_data[i] = val;
        return *this;
    }


    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::set_value(const Scalar val) {
        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());
        for (unsigned i = 0; i < total_elements; i++)
            m_data[i] = val;
        return *this;
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::set_zero() {
        return this->set_value(0);
    }

    template<typename Scalar>
    inline Tensor<Scalar>& Tensor<Scalar>::set_one() {
        return this->set_value(1);
    }


    template<typename Scalar>
    inline Scalar& Tensor<Scalar>::operator[](const std::vector<std::size_t>& idx) {
        return m_data[this->linearize_index(idx)];
    }


    template<typename Scalar>
    template<typename ...IdxPack, typename>
    inline Tensor<Scalar> Tensor<Scalar>::operator()(IdxPack ...idx) {
        std::vector<std::size_t> indices{static_cast<std::size_t>(idx)...};
        return Tensor(m_data + linearize_index(indices),
                      indices.size() == m_dims.size()
                      ? std::vector<std::size_t>{1u}
                      : std::vector<std::size_t>(m_dims.begin() + indices.size(), m_dims.end()));
    }


//TODO Doc
    template<typename Scalar>
    template<typename ...IdxPack, typename>
    inline const Tensor<Scalar> Tensor<Scalar>::operator()(IdxPack ...idx) const {
        std::vector<std::size_t> indices{static_cast<std::size_t>(idx)...};
        return Tensor(m_data + linearize_index(indices),
                      indices.size() == m_dims.size()
                      ? std::vector<std::size_t>{1u}
                      : std::vector<std::size_t>(m_dims.begin() + indices.size(), m_dims.end()));
    }

//TODO Doc ajout const
//Passe d'un vecteur d'indices a l'index correspondant dans le tableau
//m_data
    template<typename Scalar>
    inline std::size_t Tensor<Scalar>::linearize_index(const std::vector<std::size_t>& idx) const {
        assert(idx.size() <= m_dims.size());
        if (idx.size() == m_dims.size()) {
            std::size_t res = 0;
            std::size_t product = 1;
            for (std::size_t dim = m_dims.size() - 1; dim > 0; --dim) {
                res += product * idx[dim];
                product *= m_dims[dim];
            }
            res += product * idx[0];
            return res;
        } else { //moins d'indices que de dimensions
            std::vector<std::size_t> start_idx(m_dims.size(), 0);
            std::copy(idx.begin(), idx.end(), start_idx.begin());
            return linearize_index(start_idx);
        }
    }

/*
 * TODO DOC
 * Retourne l'index de l'element le plus grand du Tensor
 * L'index retourne est lineaire, c'est l'index dans m_data.
 */
    template<typename Scalar>
    inline std::size_t Tensor<Scalar>::argmax() const {
        std::size_t idx = 0;
        std::size_t size = std::accumulate(m_dims.begin(), m_dims.end(), 1u, std::multiplies<std::size_t>());
        for (unsigned i = 0; i < size; ++i)
            if (m_data[i] > m_data[idx])
                idx = i;
        return idx;
    }


/*
* Map une matrice sur le data du tenseur
* Note : On peut convertir une Eigen::Map en RMMat, mais on force dans
* ce cas une copie des donnees ; on a plus une vue sur les donnees mais
* une copie.
* C'est pourquoi le type de retour n'est pas RMMat, et pourquoi il faut
* utiliser la methode comme suit, avec auto :
*
* auto mat = mon_tenseur.as_matrix();
* (ou bien directement Eigen::Map<...>)
* Mais pas RMMat mat = ...
*/
    template<typename Scalar>
    inline Eigen::Map<RMMat < Scalar> >
    Tensor<Scalar>::as_matrix(const std::size_t rows,
                              const std::size_t cols) {
        //RMMat<Scalar> mat = Eigen::Map<RMMat<Scalar> >(m_data, rows, cols);
        return Eigen::Map<RMMat<Scalar> >(m_data, rows, cols);
    }


    template<typename Scalar>
    inline Eigen::Map<RMMat < Scalar> >

    Tensor<Scalar>::as_matrix() {
        assert(m_dims.size() == 2 && "You must specify the size of the matrix if the tensor is not rank 2");
        return Eigen::Map<RMMat<Scalar> >(m_data, m_dims[0], m_dims[1]);
    }


    template<typename Scalar>
    inline void Tensor<Scalar>::dprint() {
        std::size_t total_elements = std::accumulate(m_dims.begin(),
                                                     m_dims.end(), 1,
                                                     std::multiplies<std::size_t>());
        std::cout << "dprint : " << total_elements << " elements" << std::endl;
        for (unsigned i = 0; i < total_elements; ++i) {
            std::cout << " " << m_data[i];
        }
        std::cout << std::endl;
    }


//afficher le tenseur d'une maniere proche de numpy
    template<typename Scalar>
    inline void print(Scalar* data,
                      const std::vector<std::size_t>& dims,
                      const unsigned el_per_line,
                      const bool indent = false,
                      const unsigned spaces = 0,
                      const bool no_endl = false,
                      std::ostream& out_str = std::cout) {
        //fonction recursive
        if (dims.size() == 1) {
            /*
            * Dans le cas ou il y a une dimension a afficher
            */
            if (indent)
                out_str << std::string(" ", spaces);

            out_str << "[";
            for (unsigned i = 0; i < dims[0] - 1; ++i)
                out_str << data[i] << " ";
            out_str << data[dims[0] - 1] << "]";

            if (!no_endl)
                out_str << "," << std::endl;
        } else {
            /*
            * Dans le cas ou il reste plusieurs dimensions a afficher, on
            * reappelle la fonction avec une dimension en moins, une fois pour chaque
            * "sous-tenseur" de la premiere dimension actuelle.
            */
            if (indent)
                out_str << std::string(" ", spaces);

            out_str << "[";
            if (dims[0] != 1) {
                print(data, std::vector<std::size_t>(dims.begin() + 1, dims.end()), el_per_line / dims[1], false,
                      spaces + 1, false);
                for (unsigned i = 1; i < dims[0] - 1; ++i)
                    print(data + (i * el_per_line), std::vector<std::size_t>(dims.begin() + 1, dims.end()),
                          el_per_line / dims[1], true, spaces + 1);
                print(data + ((dims[0] - 1) * el_per_line), std::vector<std::size_t>(dims.begin() + 1, dims.end()),
                      el_per_line / dims[1], true, spaces + 1, true);
            } else
                print(data, std::vector<std::size_t>(dims.begin() + 1, dims.end()), el_per_line / dims[1], false,
                      spaces + 1, true);

            out_str << "]";

            if (!no_endl)
                out_str << "," << std::endl << std::endl;
        }
    }


    template<typename Scalar>
    inline void print(std::ostream& out_str,
                      Scalar* data,
                      const std::vector<std::size_t>& dims,
                      const unsigned el_per_line,
                      const bool indent = false,
                      const unsigned spaces = 0,
                      const bool no_endl = false) {
        print(data, dims, el_per_line, indent, spaces, no_endl, out_str);
    }


// Permet std::cout << tensor;
// Et plus generalement n'importe quel flux, comme std::fstream etc
    template<typename Scalar>
    inline std::ostream& operator<<(std::ostream& out_str, const Tensor<Scalar>& ten) {
        print(out_str,
              ten.data(),
              ten.get_dimensions(),
              std::accumulate(ten.get_dimensions().begin() + 1,
                              ten.get_dimensions().end(), 1u,
                              std::multiplies<std::size_t>()));
        return out_str;
        //TODO probleme nbr crochets
    }


    template<typename ...SizePack, typename = all_scalar_types<std::size_t, SizePack...> >
    Tensor<double> random_tensor_d(SizePack&& ...dims) {
        Tensor<double> ret(dims...);
        std::vector<std::size_t> dims_vec = ret.get_dimensions();
        std::size_t total_elements = std::accumulate(dims_vec.begin(),
                                                     dims_vec.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());

        std::default_random_engine rand_engine;
        std::uniform_real_distribution<double> distr(0, 1);
        for (std::size_t i = 0; i < total_elements; ++i)
            ret.data()[i] = distr(rand_engine);

        return ret;
    }

/*-*
 * Crée un Tensor contenant des nombres de type <code>double</code> 
 * générés aléatoirement selon une distribution normale de moyenne %%mean et d'écart-type %%stddev.
 * Les deux premiers paramètres sont le centre et l'écart-type de la distribution, et les suivants 
 * sont les dimensions du Tensor, qui doivent être convertibles 
 * en <code>std::size_t</code>, faute de quoi une erreur de compilation sera générée.
*-*/
    template<typename ...SizePack, typename = all_scalar_types<std::size_t, SizePack...> >
    Tensor<double> randn_tensor_d(const double mean, const double stddev, SizePack&& ...dims) {
        Tensor<double> ret(dims...);
        std::vector<std::size_t> dims_vec = ret.get_dimensions();
        std::size_t total_elements = std::accumulate(dims_vec.begin(),
                                                     dims_vec.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());

        for (std::size_t i = 0; i < total_elements; ++i)
            ret.data()[i] = randn(mean, stddev);

        return ret;
    }


    template<typename ReturnScalar=int, typename ...SizePack, typename = all_scalar_types<std::size_t, SizePack...> >
    Tensor<ReturnScalar> random_tensor_i(const int min, const int max, SizePack&& ...dims) {
        Tensor<ReturnScalar> ret(dims...);
        std::vector<std::size_t> dims_vec = ret.get_dimensions();
        std::size_t total_elements = std::accumulate(dims_vec.begin(),
                                                     dims_vec.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());

        //std::default_random_engine seeder;
        // les random_engine ne marchent pas correctement avec MinGW...
        // on va donc prendre un seed avec std::chrono...
        std::mt19937 rand_engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> distr(min, max);
        for (std::size_t i = 0; i < total_elements; ++i)
            ret.data()[i] = ReturnScalar(distr(rand_engine));

        return ret;
    }


//-* Retourne un %%Tensor de dimensions %%...dims ne contenant que des zéros. Les dimensions doivent être convertibles en %%std::size_t.
    template<typename ReturnScalar=int, typename ...SizePack, typename = all_scalar_types<std::size_t, SizePack...> >
    Tensor<ReturnScalar> zero_tensor(SizePack&& ...dims) {
        Tensor<ReturnScalar> ret(dims...);
        std::vector<std::size_t> dims_vec = ret.get_dimensions();
        std::size_t total_elements = std::accumulate(dims_vec.begin(),
                                                     dims_vec.end(),
                                                     1u,
                                                     std::multiplies<std::size_t>());

        ret.set_zero();

        return ret;
    }


    template<typename T1, typename T2>
    bool tensor_almost_equal(const Tensor<T1>& t1, const Tensor<T2>& t2, const double epsilon = 1e-6) {
        assert(t1.get_dimensions() == t2.get_dimensions());
        std::size_t t_size = std::accumulate(t1.get_dimensions().begin(),
                                             t1.get_dimensions().end(),
                                             1u,
                                             std::multiplies<std::size_t>());

        bool equal = true;
        for (std::size_t i = 0; i < t_size; ++i)
            if (abs(t1.data()[i] - t2.data()[i]) > epsilon)
                equal = false;
        return equal;
    }

} //namespace nnl

#endif //NNL_TENSOR_HPP
