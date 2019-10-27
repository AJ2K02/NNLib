
#ifndef NNL_CONV2D_HPP
#define NNL_CONV2D_HPP

#include "config.h"
#include "Eigen/Dense"
#include "Layer.hpp"
#include "Tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <vector>

namespace nnl {

/*
 * Cette fonction effectue la transformation d'une image en une matrice dont chaque ligne
 * est un carre de l'image qui sera a un moment recouvert parfaitement par le filtre lors
 * de la convolution. Pour plus d'information sur le resultat, chercher im2col sur google...
 */
    template<typename Derived, typename Scalar>
    void im2col(const Eigen::MatrixBase<Derived>& im,
                const unsigned im_size,
                const unsigned im_chan_px,
                const unsigned out_size,
                const unsigned ker_size,
                const unsigned nchan,
                const unsigned stride,
                const unsigned pad,
                RMMat <Scalar>& result) {
        int im_y, im_x;
        for (unsigned c = 0; c < nchan; ++c)
            for (unsigned y = 0; y < out_size; ++y) {
                im_y = y * stride - pad;
                for (unsigned x = 0; x < out_size; ++x) {
                    im_x = x * stride - pad;
                    for (unsigned ky = 0; ky < ker_size; ++ky)
                        for (unsigned kx = 0; kx < ker_size; ++kx) {
                            if (ge_zero_lt_b(im_y + ky, im_size) && ge_zero_lt_b(im_x + kx, im_size))
                                result(LINEARIZE_INDEX_3D(c, ky, kx, SQR(ker_size), ker_size),
                                       LINEARIZE_INDEX_2D(y, x, out_size))
                                        = im(0, LINEARIZE_INDEX_3D(c, im_y + ky, im_x + kx, im_chan_px, im_size));
                            else
                                result(LINEARIZE_INDEX_3D(c, ky, kx, SQR(ker_size), ker_size),
                                       LINEARIZE_INDEX_2D(y, x, out_size)) = 0;
                        }
                }
            }
    }

/*
 * Processus inverse de im2col. On part d'une matrice consideree comme le resultat
 * d'une operation im2col et on retrouve l'image initiale (*Note).
 *
 * Note : En realite, on ne retrouve pas exactement l'image originale, mais pour chaque
 * pixel de l'image, on additionne les valeurs lui correspondant dans la matrice col_arr.
 * Cela est du au fait que cette fonction est utilisee pour la retropropagation : on somme
 * donc les derivees partielles de l'erreur par rapport a chaque pixel de l'image.
 */
    template<typename Derived>
    void col2im(const RMMat <Real>& col_arr,
                const unsigned im_size,
                const unsigned out_size,
                const unsigned ker_size,
                const unsigned nchan,
                const unsigned stride,
                const unsigned pad,
                Eigen::MatrixBase<Derived>& im) {
        unsigned row, col;
        int y_im_pos, x_im_pos;
        for (unsigned c = 0; c < nchan; ++c)
            for (unsigned y = 0; y < out_size; ++y) {
                y_im_pos = y * stride - pad;
                for (unsigned x = 0; x < out_size; ++x) {
                    x_im_pos = x * stride - pad;
                    col = LINEARIZE_INDEX_2D(y, x, out_size);
                    for (unsigned ky = 0; ky < ker_size; ++ky)
                        for (unsigned kx = 0; kx < ker_size; ++kx) {
                            if (ge_zero_lt_b(y_im_pos + ky, im_size) && ge_zero_lt_b(x_im_pos + kx, im_size)) {
                                row = LINEARIZE_INDEX_3D(c, ky, kx, SQR(ker_size), ker_size);
                                im(0, LINEARIZE_INDEX_3D(c, y_im_pos + ky, x_im_pos + kx, SQR(im_size), im_size))
                                        += col_arr(row, col);
                            }
                        }
                }
            }
    }

/*
 * Fonction permettant de faire une rotation de 180 degres d'une matrice carre,
 * normalement un filtre. Cette operation est utilisee pour la retro-propagation
 * de la convolution, mais je ne l'utilise peut-etre plus...
 */
    RMMat <Real> rotate_kernels180(const RMMat <Real>& ker) {
        unsigned nb_coeff = ker.cols();
        RMMat<Real> rotated(ker.rows(), nb_coeff);
        for (std::size_t k = 0; k < ker.rows(); ++k)
            for (std::size_t i = 0; i < nb_coeff; ++i)
                rotated(k, nb_coeff - 1 - i) = ker(k, i);
        return rotated;
    }

/*
 * Equivalent a la fonction im2col, mais avec une grande boucle au lieu de beaucoup
 * de petites boucles imbriquees. Il est necessaire d'avoir une matrice "indices", calculee
 * avec la fonction im2col initiale. Pour voir un exemple d'utilisation de cette fonction,
 * voir le constructeur de la classe Conv2d.
 *
 * Fait interessant : en compilant avec -O3, le gain de vitesse de cette fonction
 * par rapport a im2col est non-existant...
 */
    void im2col_with_indices(const RMMat <std::size_t>& indices, const RMMat <Real>& img, RMMat <Real>& out) {
        assert(img.rows() == 1);
        //Malheureusement la version 3.4 d'Eigen n'est pas encore stable,
        //sans quoi on pourrait utiliser le "custom indexing"
        for (std::size_t i = 0; i < out.rows() * out.cols(); ++i)
            out.data()[i] = img(0, indices.data()[i]);
    }

/*
 * Equivalent a la fonction col2im, mais avec une grande boucle au lieu de beaucoup
 * de petites boucles imbriquees. Il est necessaire d'avoir une matrice "indices", calculee
 * avec la fonction im2col (et non pas col2im!).
 *
 * La matrice indices doit contenir uniquement des entiers positifs...
 */
    template<typename ScalarInd, typename ScalarCol, typename DerivedIm>
    void col2im_with_indices(const RMMat <ScalarInd>& indices, const RMMat <ScalarCol>& col_arr,
                             Eigen::MatrixBase<DerivedIm>& out) {
        for (std::size_t i = 0; i < col_arr.cols() * col_arr.rows(); ++i)
            out(static_cast<std::size_t>(indices.data()[i])) += col_arr.data()[i];
    }


/*
 * Classe representant une couche convolutive. L'implementation utilise des fonctions
 * im2col et col2im afin d'optimiser les performances. Pour une plus ample description
 * de la classe, voir la page de documentation qui lui est dediee.
 */
    template<typename Activation>
    class Conv2d : public Layer {
    public:
        Conv2d() = delete;

        //-* Constructeur par defaut
        Conv2d(const Conv2d& c) = default;

        /*-*
         * Constructeur prenant les hyper-parametres. Il est cense etre utilise lors de la creation des couches
         * pour le reseau neuronal.<br>
         * Parametres : <br>
         * %%in_s : taille d'un cote des images d'entree<br>
         * %%in_c : nombre de canaux des images d'entree<br>
         * %%ker_s : taille d'un cote des noyaux<br>
         * %%out_c : nombre de canaux des images de sorties<br>
         * %%stride : pas<br>
         * %%padding : padding, soit nombre de couches de zeros rajoutees autour des images avant la convolution.
         *-*/
        Conv2d(unsigned in_s, unsigned in_c,
               unsigned ker_s, unsigned out_c,
               unsigned stride, unsigned padding);

        ~Conv2d() = default;

        //-* Permet d'initialiser les poids et les biais selon une classe d'initialisation passee par le parametre template %%Init.
        template<typename Init>
        Conv2d init();

        //-* Retourne le nom de la couche (utilise pour sauvegarder le reseau de neurone).
        std::string get_name() const {
            return "Conv2d{" + std::to_string(m_im_chan) + ',' +
                   std::to_string(m_im_size) + ',' + std::to_string(m_out_chan) + ',' +
                   std::to_string(m_ker_size) + ',' + std::to_string(m_lr) + ',' +
                   std::to_string(m_stride) + ',' + std::to_string(m_padding) + ',' +
                   Activation::get_name() + '}';
        };

        //-* Retourne la matrice de poids (= noyaux).
        RMMat <Real> get_weights() const override { return m_kernel; }

        //-* Permet de changer la matrice de poids (= noyaux).
        void set_weight(const RMMat <Real>& w) { m_kernel = w; }

        //-* Retourne la matrice du gradient de l'erreur par rapport aux poids (aux noyaux).
        RMMat <Real> get_delta_w() const override { return dl_dw; }

        //-* Retourne la matrice d'unites de biais.
        RMMat <Real> get_biases() { return m_bias; }

        //-* Permet de changer la matrice d'unites de biais.
        RMMat <Real> set_biases(const RMMat <Real>& bias) { m_bias = bias; }

        //-* Retourne la matrice du gradient de l'erreur par rapport aux unites de biais.
        RMMat <Real> get_delta_b() { return dl_db; }

        //-* Retourne le taux d'apprentissage de la couche.
        Real get_lr() const { return m_lr; }

        //-* Permet de definir le taux d'apprentissage de la couche.
        void set_lr(const Real lr) { m_lr = lr; }

        //-* Retourne la matrice de pre-activation resultant de la derniere utilisation de la couche.
        Matrix get_pre_act() const override { return m_z; }

        //-* Retourne la matrice de sortie de la derniere utilisation de la couche.
        Matrix get_output() const override { return m_out; }

        //-* Retourne le nombre de valeurs attendues en entree de la couche.
        unsigned get_inputs_nb() const override { return m_im_chan_size * m_im_chan; }

        //-* Retourne le nombre de valeurs attendues en entree de la couche.
        unsigned get_outputs_nb() const override { return m_out_chan_size * m_out_chan; }

        //-* Effectue la propagation avant (Note : utiliser la version prenant un tenseur en parametre).
        RMMat <Real> forward(const RMMat <Real>& inputs) override;

        //-* Effectue la propagation avant.
        Tensor<Real> forward(Tensor<Real>& inputs);

        /*-*
         * Effectue la retro-propagation.<br>
         * %% dummy : ce parametre n'est pas utilise mais necessaire pour respecter l'interface de la classe parent Layer
         * %% dl_dy : gradient de l'erreur par rapport a la sortie de la couche.
         *-*/
        RMMat <Real> backward(const RMMat <Real>& dummy, const RMMat <Real>& dl_dy) override;

        //-* Met a jour les parametres en utilisant les donnes presentes dans les matrices de gradient.
        void update_param() override;

        //-* Ecrit les parametres de la couche dans un flux
        void print_parameters(std::ostream& s) const;

        //-* Charge les parametres a partir d'un flux
        void load_parameters(std::ifstream& file);

        //-* Verifie que les hyper-parametres correspondent a la ligne chargee depuis le fichier
        void check_hyper_parameters(const std::string& line) const;

        //-* Charge les hyper-parametres necessaires depuis la ligne du fichier.
        void load_hyper_parameters(const std::string& line);

    private:
        unsigned m_im_size, m_out_size;
        unsigned m_im_chan_size, m_out_chan_size;
        unsigned m_im_chan, m_out_chan;
        unsigned m_ker_size;
        unsigned m_stride;
        unsigned m_padding;
        Real m_lr;
        RMMat <Real> m_kernel;
        RMMat <Real> m_bias;

        RMMat <Real> m_convolved;
        std::vector<RMMat < Real>> m_col_im;
        RMMat <Real> m_z;
        RMMat <Real> m_out;

        // Donnes pour la retro-propagation
        RMMat <Real> dl_dz;
        RMMat <Real> dl_dw;
        RMMat <Real> dl_db;
        RMMat <Real> dl_dx;

        RMMat <Real> m_im2col_ind;
    };


    template<typename Activation>
    Conv2d<Activation>::Conv2d(const unsigned in_s, const unsigned in_c, const unsigned ker_s, const unsigned out_c,
                               const unsigned stride, const unsigned padding)
            : m_im_size(in_s), m_im_chan_size(SQR(in_s)), m_im_chan(in_c),
              m_out_size((in_s + 2 * padding - ker_s) / stride + 1), m_out_chan_size(SQR(m_out_size)),
              m_out_chan(out_c),
              m_ker_size(ker_s),
              m_stride(stride), m_padding(padding) {
        // Initialiser les poids et les biais
        m_bias = RMMat<Real>::Zero(out_c, 1);
        m_kernel.resize(m_out_chan, SQR(m_ker_size) * m_im_chan);
        double stddev = std::sqrt(2.0 / (m_im_chan * m_im_chan_size + m_out_chan * m_out_chan_size));
        for (std::size_t i = 0; i < m_kernel.rows() * m_kernel.cols(); ++i)
            m_kernel.data()[i] = randn(0, stddev);

        dl_dw.resize(m_kernel.rows(), m_kernel.cols());
        dl_db.resize(m_out_chan, 1);

        m_im2col_ind.resize(SQR(m_ker_size) * m_im_chan, m_out_chan_size);
        RMMat<Real> temp(1, m_im_chan * m_im_chan_size);
        temp.row(0).setLinSpaced(temp.cols(), 0, temp.cols() - 1);
        im2col(temp, m_im_size, m_im_chan_size,
               m_out_size, m_ker_size, m_im_chan, m_stride, m_padding, m_im2col_ind);
    }

    template<typename Activation>
    template<typename Init>
    Conv2d<Activation> Conv2d<Activation>::init() {
        Init::initialize_weight(m_kernel, m_kernel.cols(), m_out_chan * m_out_chan_size);
        Init::initialize_bias(m_bias, m_kernel.cols(), m_out_chan * m_out_chan_size);
        return *this;
    }


    template<typename Activation>
    RMMat <Real> Conv2d<Activation>::forward(const RMMat <Real>& inputs) {
        //resize m_col_im si besoin
        if (m_col_im.size() != inputs.rows()) {
            const unsigned cols = m_out_chan_size;
            const unsigned rows = SQR(m_ker_size) * m_im_chan;
            m_col_im.resize(inputs.rows(), RMMat<Real>::Zero(rows, cols));
        }

        m_z.resize(inputs.rows(), m_out_chan_size * m_out_chan);
        for (unsigned i = 0; i < inputs.rows(); ++i) {
            im2col(inputs.row(i), m_im_size, m_im_chan_size,
                   m_out_size, m_ker_size, m_im_chan, m_stride, m_padding, m_col_im[i]);
            //im2col_with_indices(m_im2col_ind, inputs.row(i), m_col_im[i]);
            // Effectuer la convolution
            m_convolved = (m_kernel * m_col_im[i]).eval();
            //TODO Exemple ou utiliser MKL aurait ete plus simple + efficace...
            std::move(m_convolved.data(),
                      m_convolved.data() + m_convolved.cols() * m_convolved.rows(),
                      m_z.data() + i * m_z.cols());
        }
        // Ajout des biais
        for (unsigned k = 0; k < m_out_chan; ++k)
            m_z.block(0, k * m_out_chan_size, m_z.rows(), m_out_chan_size).array() += m_bias(k, 0);
        // Activation
        m_out = m_z.unaryExpr(std::ref(Activation::forward));
        return m_out;
    }

    template<typename Activation>
    Tensor<Real> Conv2d<Activation>::forward(Tensor<Real>& inputs) {
        //inputs doit etre de dimension 4: nb_img X nb_canaux X hauteur X largeur
        std::vector<std::size_t> dims = inputs.get_dimensions();
        auto in_mat = inputs.as_matrix(dims[0], dims[1] * dims[2] * dims[3]);
        return Tensor<Real>(forward(in_mat)).reshape(dims[0], m_out_chan, m_out_size, m_out_size);
    }

// Calcul des gradients pour la retro-propagation.
// Note : dl_dy n'est pas const a cause de l'utilisation
// de Eigen::Map pour le redimensionner. Il n'y a
// (actuellement, 09.2019) pas
// moyen d'indiquer au compilateur que l'objet Eigen::Map
// ne modifiera pas les donnees de dl_dy...
    template<typename Activation>
    RMMat <Real> Conv2d<Activation>::backward(const RMMat <Real>& dummy, const RMMat <Real>& dl_dy) {
        dl_dz = dl_dy.cwiseProduct(m_z.unaryExpr(std::ref(Activation::backward)));
        dl_dw.setZero();
        dl_dx.resize(dl_dz.rows(), m_im_chan_size * m_im_chan);
        dl_dx.setZero();
        for (std::size_t i = 0; i < dl_dz.rows(); ++i) {
            //TODO : 1 seul reshape
            Eigen::Map<RMMat<Real>> reshaped_dl_dz(dl_dz.row(i).data(), m_out_chan, m_out_chan_size);
            dl_dw += reshaped_dl_dz * m_col_im[i].transpose();
            // On passe par une variable row car la methode row() retourne un objet temporaire qui ne peut pas
            // etre pris par ref non constante dans la fonction col2im tel quel.
            auto row = dl_dx.row(i);
            col2im(m_kernel.transpose() * reshaped_dl_dz, m_im_size, m_out_size, m_ker_size, m_im_chan, m_stride,
                   m_padding, row);
            //col2im_with_indices(m_im2col_ind, m_kernel.transpose() * reshaped_dl_dz, row);
        }
        // Biais :
        dl_db.setZero();
        for (unsigned c = 0; c < m_out_chan; ++c)
            dl_db(c, 0) = dl_dz.block(0, c * m_out_chan_size, dl_dz.rows(), m_out_chan_size).sum();
        return dl_dx;
    }

    template<typename Activation>
    void Conv2d<Activation>::update_param() {
        // Biais :
        m_bias -= m_lr * dl_db;
        // Noyau :
        m_kernel -= m_lr * dl_dw;
    }

    template<typename Activation>
    void Conv2d<Activation>::print_parameters(std::ostream& s) const {
        // noyaux
        for (unsigned i = 0; i < m_kernel.rows() * m_kernel.cols(); ++i)
            s << m_kernel.data()[i] << ' ';
        // biais
        for (unsigned i = 0; i < m_bias.rows(); ++i)
            s << m_bias.data()[i] << ' ';
        s << std::endl;
    }

    template<typename Activation>
    void Conv2d<Activation>::load_parameters(std::ifstream& file) {
        //noyaux
        for (unsigned i = 0; i < m_kernel.rows() * m_kernel.cols(); ++i)
            file >> m_kernel.data()[i];
        //TODO Trop lent, tout lire d'un coup
        //biais
        for (unsigned i = 0; i < m_bias.rows(); ++i)
            file >> m_bias.data()[i];
        file.ignore(); // passer le '\n'
    }

    template<typename Activation>
    void Conv2d<Activation>::load_hyper_parameters(const std::string& line) {
        // stringstream contenant la ligne a partir du premier hyper parametre
        std::stringstream ss(std::string(std::find(line.begin(), line.end(), '{') + 1, line.end()));
        int dump;
        ss >> dump;
        ss.ignore();
        ss >> dump;
        ss.ignore();
        ss >> dump;
        ss.ignore();
        ss >> dump;
        ss.ignore();
        // Seul le learning rate nous interesse
        ss >> m_lr;
    }

    template<typename Activation>
    void Conv2d<Activation>::check_hyper_parameters(const std::string& line) const {
        // stringstream contenant la ligne a partir du premier hyper parametre
        std::stringstream ss(std::string(std::find(line.begin(), line.end(), '{') + 1, line.end()));
        unsigned in_c, in_s, out_c, ker_s, stride, pad;
        double lr;
        ss >> in_c;
        ss.ignore();
        ss >> in_s;
        ss.ignore();
        ss >> out_c;
        ss.ignore();
        ss >> ker_s;
        ss.ignore();
        ss >> lr;
        ss.ignore();
        ss >> stride;
        ss.ignore();
        ss >> pad;
        assert(m_im_chan == in_c);
        assert(m_im_size == in_s);
        assert(m_out_chan == out_c);
        assert(m_ker_size == ker_s);
        assert(m_lr == lr);
        assert(m_stride == stride);
        assert(m_padding == pad);
    }

} // namespace nnl

#endif //NNL_CONV2D_HPP
