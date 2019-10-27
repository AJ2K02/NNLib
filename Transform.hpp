
#ifndef NNL_TRANSFORM_HPP
#define NNL_TRANSFORM_HPP

#include <algorithm>
#include <iostream>

#include "Tensor.hpp"

namespace nnl {

    struct Transform {
        //-* Méthode servant à encoder les valeurs du %%Tensor %%x. En d'autres termes, cette méthode effectue la transformation.
        inline virtual void encode(Tensor <Real>& x) { std::cerr << "Method encode not overriden." << std::endl; }

        /*-*
         * Méthode servant à décoder les valeurs du Tensor x. En d'autres termes, cette méthode "annule" la transformation qui a été effectuée.
         * Il n'est pas nécessaire d'implémenter cette méthode dans la plupart des cas. Par exemple, si vous souhaitez simplement effectuer
         * des transformations sur les données avant l'apprentissage, cette méthode est inutile.
        *-*/
        inline virtual void decode(Tensor <Real>& x) { std::cerr << "Method decode not overriden." << std::endl; }
    };


/*
 * Change les dimensions du Tensor, permettant par exemple de passer d'un tenseur (10000 x 1200) a
 * un tenseur (10000 x 3 x 20 x 20).
 */
    template<int... args>
    struct Reshape : public Transform {
        inline void encode(Tensor <Real>& x) {
            initial_dims = x.get_dimensions();
            x.reshape_in_place_no_return(args...);
        }

        //TODO inline Tensor<Real> decode(const Tensor<Real>& x) {}

        std::vector<std::size_t> initial_dims;
    };


    struct Batch : public Transform {
        Batch(const unsigned _batch_size) : m_bs(_batch_size) {}

        inline void encode(Tensor <Real>& x) {
            std::vector<std::size_t> new_dims{x.get_dimensions()[0] / m_bs, m_bs};
            //std::copy(x.get_dimensions().begin() + 1, x.get_dimensions().end(), std::back_inserter(new_dims));
            new_dims.insert(new_dims.end(), x.get_dimensions().begin() + 1, x.get_dimensions().end());
            x.reshape_in_place_no_return(new_dims);
        }

        unsigned m_bs;
    };


/*
 * Divise tout les nombres par Dividend.
 * Utile par exemple pour les images, pour que les pixels passent de [0, 255] a [0, 1]
 */
    struct Divide : public Transform {
        Divide(const Real _d) : m_d(_d) {};

        inline void encode(Tensor <Real>& x) {
            //return x / m_d;
            x /= m_d;
        }

        inline void decode(Tensor <Real>& x) {
            x *= m_d;
        }

        Real m_d;
    };


/*
 * Multiplie chaque sous tenseur d'input par un facteur de maniere
 * a ce qu'ils aient chacun un ecart type de 1.
 */
    struct Standardize : public Transform {
        inline void encode(Tensor <Real>& x) {
            m_mean = x.mean();
            m_stddev = x.std(m_mean);
            // m_eps permet d'eviter les divisions par zero...
            x -= m_mean;
            x /= m_stddev + m_eps;
            //x = (x - m_mean) / (m_stddev+m_eps);
        }

        inline void decode(Tensor <Real>& x) {
            x *= (m_stddev + m_eps);
            x += m_mean;
        }

        Real m_mean;
        Real m_stddev;
        static constexpr Real m_eps = 1e-8;
    };


    struct ToOneHot : public Transform {
        ToOneHot(const unsigned _classes) : m_num_classes(_classes) {}

        inline void encode(Tensor <Real>& x) {
            std::size_t size;
            if (x.get_dimensions().size() == 1)
                size = x.get_dimensions()[0];
            else if (x.get_dimensions().size() == 2)
                size = x.get_dimensions()[0] * x.get_dimensions()[1];
            else
                assert("Input tensor must be of rank 1 or 2." && 0);

            Tensor<Real> r = x.reshape(size);
            //Tensor<Real> oh(size, m_num_classes);
            //Tensor<Real> zeros = zero_tensor<Real>(m_num_classes);
            auto oh = zero_tensor<Real>(size, m_num_classes);
            for (std::size_t i = 0; i < size; ++i) {
                //zeros(r(i).item()).item() = 1;
                //oh(i).copy_data(zeros);
                //zeros(r(i).item()).item() = 0;
                
                oh(i, r(i).item()).item() = 1;
            }
            x = oh;
        }

        inline Tensor <Real> decode(const Tensor <Real>& x) {
            std::cerr << "Not Implemented yet." << std::endl;
        }

        unsigned m_num_classes;
    };

} //namespace nnl

#endif //NNL_TRANSFORM_HPP
