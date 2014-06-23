#ifndef Esnesnesn_fechoActivations_h
#define Esnesnesn_fechoActivations_h
//#include "fechoMath.h"
#include "armadillo"
using namespace arma;

namespace Fecho {
    
    template <typename T>
    class ActivationFunctionBase {
    public:
        ActivationFunctionBase() {}
        virtual void process(Col<T> &x) {};
        virtual void invProcess(Mat<T> &x) {};
    };
    
    template <typename T>
    class ActivationFunctionLinear : public ActivationFunctionBase<T> {
    public:
        inline void process(Col<T> &x) {};
        inline void invProcess(Mat<T> &x) {};
    };
    
    
    template <typename T>
    class ActivationFunctionTanh : public ActivationFunctionBase<T> {
    public:
        inline void process(Col<T> &x) {
            x = tanh(x);
        };
        inline void invProcess(Mat<T> &x) {
            x = atanh(x);
        };
    };


    template <typename T>
    class ActivationFunctionSigmoid : public ActivationFunctionBase<T> {
    public:
        inline void process(Col<T> &x) {
            x = 1.0 / (1.0 + exp(-x));
            
        }
        inline void invProcess(Mat<T> &x) {
            x = -log((1.0/x) - 1.0);
//            for(int i=0; i < x.n_elem; i++) {
//                x[i] = x[i]==0 ? 0 : 1.0 / x[i];
//            }
//            x = log(x - 1.0);
        }
    };
}

#endif
