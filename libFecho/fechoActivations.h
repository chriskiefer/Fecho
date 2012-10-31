#ifndef Esnesnesn_fechoActivations_h
#define Esnesnesn_fechoActivations_h
#include "fechoMath.h"

namespace Fecho {
    
    template <typename T>
    class ActivationFunctionBase {
    public:
        ActivationFunctionBase() {}
        virtual void process(T *x, const int size) {};
        virtual void invProcess(T* x, const int size) {};
    };
    
    template <typename T>
    class ActivationFunctionLinear : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {};
        inline void invProcess(T* x, const int size) {};
    };
    
    
    template <typename T>
    class ActivationFunctionTanh : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = tanh(x[i]);
#else
            Math::vvtanh(x, x, &size);
#endif
        };
        inline void invProcess(T* x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = atanh(x[i]);
#else
            Math::vvatanh(x, x, &size);
#endif
        };
    };
    

    template <typename T>
    class ActivationFunctionSigmoid : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = 1.0 / (1.0 + exp(x[i]));
#else
            Math::vvexp(x, x, &size);
            float one = 1.0f;
            Math::vsadd(x, 1, &one, x, 1, size);
            Math::vvrec(x, x, &size);
#endif
        }
        inline void invProcess(T* x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = log(1.0/x[i] - 1.0);
#else
            Math::vvrec(x, x, &size);
            float minusone = -1.0f;
            Math::vsadd(x, 1, &minusone, x, 1, size);
            Math::vvlog(x, x, &size);
#endif
        }
    };
}

#endif
