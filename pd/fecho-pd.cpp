#include "m_pd.h"
#include <cmath>

#include "fecho.hpp"
#include <Accelerate/Accelerate.h>
#include <fstream>
#include "armadillo"

static t_class *fecho_pd_class;

using namespace Fecho;
typedef PD_FLOATTYPE FECHOTYPE;

class fecho_pd {
public:
    
    int testint;
    
    typedef struct _fecho_pd {
        t_object  x_obj;
        fecho_pd *f;
        t_outlet *out0;
        t_outlet *out1;
        t_outlet *out2;
        t_atom *activationslist, *outputslist, *errorslist;
        t_int nIns, nRes, nOuts;
    } t_fecho_pd;


    ActivationFunctionTanh<FECHOTYPE> resAct;
    Reservoir<FECHOTYPE> net;
    ActivationFunctionLinear<FECHOTYPE> roAct;
    ReadOut<FECHOTYPE> ro;
    ReadOutInitialiser<FECHOTYPE> roInit;
    Simulator<FECHOTYPE> sim;
    bool needToInitialise;
    FECHOTYPE resRangeLow, resRangeHigh, resConn;
    FECHOTYPE inRangeLow, inRangeHigh, inConn;
    FECHOTYPE fbRangeLow, fbRangeHigh, fbConn;
    FECHOTYPE alpha;
    Col<FECHOTYPE> inputVec;
    int washout;
    FECHOTYPE resScale;
    
    enum trainTypes {LS, PI, RR} trainType;
    
    vector<t_symbol*> trainInNames, trainOutNames;
    
    fecho_pd(t_fecho_pd *x) {
        net = Reservoir<FECHOTYPE>(x->nIns, x->nRes, &resAct);
        ro = ReadOut<FECHOTYPE> (net, x->nOuts, &roAct);
        ro.setMapInsToOuts(true);
        sim = Simulator<FECHOTYPE>(net, ro);
        resRangeLow = -0.1;
        resRangeHigh = 0.1;
        resConn = 0.5;
        inRangeLow = -0.5;
        inRangeHigh = 0.5;
        inConn = 1.0;
        fbConn = 0.0;
        fbRangeLow = -1.0;
        fbRangeHigh = 1.0;
        alpha = 0.9;
        washout = 3;
        resScale = 1.0;
        trainType = RR;
        init(x);
    }

    static void bang_proxy(t_fecho_pd *x) {
        x->f->bang(x);
    }

    void bang(t_fecho_pd *x) {
        runOneEpoch(x);
    }
    
    void runOneEpoch(t_fecho_pd *x) {
        if (needToInitialise) {
            init(x);
        }
        sim.simulate(inputVec);
        Col<FECHOTYPE> res = ro.getOutputs();
        for(int i=0; i < x->nOuts; i++) {
            x->outputslist[i].a_w.w_float = ro.getOutputs().at(i);
        }
        outlet_list(x->out0, &s_list, x->nOuts, x->outputslist);
    }
    
    static void dumpActs_proxy(t_fecho_pd *x) {x->f->dumpActs(x);}
    void dumpActs(t_fecho_pd *x) {
        for(int i=0; i < x->nRes; i++) {
            x->activationslist[i].a_w.w_float = net.getActivations().at(i);
        }
        outlet_list(x->out1, &s_list, x->nRes, x->activationslist);
    }

    static void reset_proxy(t_fecho_pd *x) {x->f->reset(x);}
    void reset(t_fecho_pd *x) {
        net.resetActivations();
    }

    static void randomise_proxy(t_fecho_pd *x) {x->f->randomise(x);}
    void randomise(t_fecho_pd *x) {
        net.randomiseActivations();
    }
    
    bool listTo3Floats(t_fecho_pd *x, int argcount, t_atom *argvec, float &f1, float &f2, float &f3) {
        bool argError = false;
        if (argcount == 3) {
            if (argvec[0].a_type == A_FLOAT && argvec[1].a_type == A_FLOAT && argvec[2].a_type == A_FLOAT) {
                f1 = argvec[0].a_w.w_float;
                f2 = argvec[1].a_w.w_float;
                f3 = argvec[2].a_w.w_float;
            }else{
                argError = true;
            }
        }else{
            if (argcount != 0) {
                argError = true;
            }
        }
        if (argError) {
            pd_error(x, "Three floating point arguments are required for this message:\n1. Low range\n2. High range\n3. Connectivity");
        }
        return argError;
    }
    
    static void resparams_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->resparams(x, selector, argcount, argvec);}
    void resparams(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        if (!listTo3Floats(x, argcount, argvec, resRangeLow, resRangeHigh, resConn)) {
            needToInitialise = true;
        }
        post("Resp: %f %f %f", resRangeLow, resRangeHigh, resConn);
    }

    static void inputparams_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->inputparams(x, selector, argcount, argvec);}
    void inputparams(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        if (!listTo3Floats(x, argcount, argvec, inRangeLow, inRangeHigh, inConn)) {
            needToInitialise = true;
        }
        post("insp: %f %f %f", inRangeLow, inRangeHigh, inConn);
    }

    static void fbparams_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->fbparams(x, selector, argcount, argvec);}
    void fbparams(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        if (!listTo3Floats(x, argcount, argvec, fbRangeLow, fbRangeHigh, fbConn)) {
            needToInitialise = true;
        }
        post("fbp: %f %f %f", fbRangeLow, fbRangeHigh, fbConn);
    }

    static void spectralRadius_proxy(t_fecho_pd *x, t_float alpha) {x->f->spectralRadius(x, alpha);}
    void spectralRadius(t_fecho_pd *x, t_float newalpha) {
        alpha = newalpha;
        needToInitialise = true;
        post("sr: %f", alpha);
    }

    static void washout_proxy(t_fecho_pd *x, t_float w) {x->f->setWashout(x, w);}
    void setWashout(t_fecho_pd *x, t_float newWashout) {
        washout = static_cast<int>(newWashout);
        post("washout: %f", washout);
    }

    static void resScale_proxy(t_fecho_pd *x, t_float w) {x->f->setResScale(x, w);}
    void setResScale(t_fecho_pd *x, t_float newScale) {
        net.scaleReservoir(newScale);
        post("scale: %f", newScale);
    }

    static void trainType_proxy(t_fecho_pd *x, t_symbol *w) {x->f->setTrainType(x, w);}
    void setTrainType(t_fecho_pd *x, t_symbol *tt) {
        post("trainType: %s", tt->s_name);
        string newTrainType(tt->s_name);
        if (newTrainType == "leastSquares") {
            trainType = LS;
        }else if (newTrainType == "pseudoInverse") {
            trainType = PI;
        }else if (newTrainType == "ridgeRegression") {
            trainType = RR;
        }else{
            post("Train type not recognised, try 'leastSquares', 'pseudoInverse', or 'ridgeRegression'");
        }
    }

    static void trainInBuffer_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->trainInBuffer(x, selector, argcount, argvec);}
    void trainInBuffer(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        attachBuffer(x, selector, argcount, argvec, x->nIns, trainInNames);
    }

    static void trainOutBuffer_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->trainOutBuffer(x, selector, argcount, argvec);}
    void trainOutBuffer(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        attachBuffer(x, selector, argcount, argvec, x->nOuts, trainOutNames);
    }

    void attachBuffer(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec, int maxIndex, vector<t_symbol*> &bufNames) {
        post("attachBuffer: %d", argcount);
        if (argcount == 2) {
            if (argvec[0].a_type == A_FLOAT && argvec[1].a_type == A_SYMBOL) {
                post("args ok");
                int idx = static_cast<int>(argvec[0].a_w.w_float);
                t_symbol *bufname = argvec[1].a_w.w_symbol;
                if (idx < maxIndex) {
                    bufNames[idx] = bufname;
                    post("args: %d %s", idx, bufname->s_name);
                }else{
                    pd_error(x, "Fecho: index out of range");
                }
            }else{
                pd_error(x, "Fecho: I need an index and an array name");
            }
        }else{
            pd_error(x, "Fecho: Two arguments required: index, array name");
        }
    }

    static void train_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->train(x, selector, argcount, argvec);}
    void train(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        if (needToInitialise) {
            init(x);
        }
        bool everythingOK = true;
        t_garray *inArrays[x->nIns];
        t_garray *outArrays[x->nOuts];
        float *inputArray[x->nIns];
        float *outputArray[x->nOuts];
        int inCount[x->nIns];
        int outCount[x->nOuts];
        post("Train");
        post("Inputs: ");
        for(int i=0; i < x->nIns; i++) {
            post("[%d] %s", i, trainInNames[i] ? trainInNames[i]->s_name : "");
            if (trainInNames[i] == NULL) {
                everythingOK = false;
                pd_error(x, "Please select a training array for this input");
            }else{
                inArrays[i] = (t_garray*)pd_findbyclass(trainInNames[i], garray_class);
                if (!inArrays[i]) {
                    pd_error(x, "Array not found");
                    everythingOK = false;
                }else{
                    bool ok = garray_getfloatarray(inArrays[i], &inCount[i], &inputArray[i]);
                    if (!ok) {
                        pd_error(x, "Bad array");
                        everythingOK = false;
                    }
                }
            }
        }
        post("Outputs: ");
        for(int i=0; i < x->nOuts; i++) {
            post("[%d] %s", i, trainOutNames[i] ? trainOutNames[i]->s_name : "");
            if (trainOutNames[i] == NULL) {
                everythingOK = false;
                pd_error(x, "Please select a training array for this output");
            }else{
                outArrays[i] = (t_garray*)pd_findbyclass(trainOutNames[i], garray_class);
                if (!outArrays[i]) {
                    pd_error(x, "Array not found");
                    everythingOK = false;
                }else{
                    bool ok = garray_getfloatarray(outArrays[i], &outCount[i], &outputArray[i]);
                    if (!ok) {
                        pd_error(x, "Bad array");
                        everythingOK = false;
                    }
                }
            }
        }
        if(!everythingOK) {
            pd_error(x, "Errors in training configuration");
        }else{
            int firstInCount = 0, firstOutCount = 0;
            bool countsAreTheSame = true;
            if (x->nIns > 0) {
                firstInCount = inCount[0];
                for(int i=1; i < x->nIns; i++) {
                    if(inCount[i] != firstInCount) {
                        countsAreTheSame = false;
                    }
                }
            }
            firstOutCount = outCount[0];
            for(int i=1; i < x->nOuts; i++) {
                if (outCount[i] != firstOutCount) {
                    countsAreTheSame = false;
                }
            }
            if (countsAreTheSame) {
                countsAreTheSame = firstInCount == firstOutCount;
            }
            if(!countsAreTheSame) {
                pd_error(x, "The training arrays should all be the same size");
            }else{
                post("Training with %d examples", firstOutCount);
                Mat<FECHOTYPE> trainIn;
                trainIn.set_size(firstOutCount, x->nIns);
                if (x->nIns > 0) {
                    for(int i=0; i < firstOutCount; i++) {
                        for(int j=0; j < x->nIns; j++) {
                            trainIn(i, j) = inputArray[j][i];
                            
                        }
                    }
                }
                Mat<FECHOTYPE> trainOut;
                trainOut.set_size(firstOutCount, x->nOuts);
                for(int i=0; i < firstOutCount; i++) {
                    for(int j=0; j < x->nOuts; j++) {
                        trainOut(i, j) = outputArray[j][i];
                    }
                }
                
                switch(trainType) {
                    case LS:
                    {
                        TrainerLeastSquares<FECHOTYPE> lstrainer(&sim, &ro, trainIn, trainOut, washout);
                        lstrainer.train();
                        break;
                    }
                    case PI:
                    {
                        TrainerPseudoInverse<FECHOTYPE> pitrainer(&sim, &ro, trainIn, trainOut, washout);
                        pitrainer.train();
                        break;
                    }
                    case RR:
                    {
                        TrainerRidgeRegression<FECHOTYPE> rrtrainer(&sim, &ro, trainIn, trainOut, washout, 0.1);
                        rrtrainer.train();
                        break;
                    }
                        
                };
                
                //get training error
                Mat<FECHOTYPE> testOut;
                testOut.set_size(firstOutCount, x->nOuts);
                for(int i=0; i < firstOutCount; i++) {
                    for(int j=0; j < x->nIns; j++) {
                        inputVec(j) = trainIn(i,j);
                    }
                    sim.simulate(inputVec);
                    Col<FECHOTYPE> res = ro.getOutputs();
                    for(int j=0; j < x->nOuts; j++) {
                        testOut(i,j) = ro.getOutputs().at(j);
                    }
                }
                
                for(int i=0; i < x->nOuts; i++) {
                    Col<FECHOTYPE> tIn = trainOut.col(i).rows(washout, trainOut.n_rows-1);
                    Col<FECHOTYPE> tOut = testOut.col(i).rows(washout, trainOut.n_rows-1);
                    FECHOTYPE nmrse = error<FECHOTYPE>::NMRSE(tIn, tOut);
                    x->errorslist[i].a_w.w_float = nmrse;
                }
                
                outlet_list(x->out2, &s_list, x->nOuts, x->errorslist);


            }
        }
    }

    
    static void listInput_proxy(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {x->f->listInput(x, selector, argcount, argvec);}
    void listInput(t_fecho_pd *x, t_symbol *selector, int argcount, t_atom *argvec) {
        if (argcount == x->nIns) {
            bool typeError = false;
            for(int i=0; i < x->nIns; i++) {
                if (argvec[i].a_type == A_FLOAT) {
                    inputVec[i] = argvec[i].a_w.w_float;
                }else{
                    typeError = true;
                    break;
                }
            }
            if (typeError) {
                pd_error(x, "All inputs must be numbers");
            }else{
                runOneEpoch(x);
            }
        }else{
            pd_error(x, "fecho: %ld inputs are required", x->nIns);
        }
    }

    static void init_proxy(t_fecho_pd *x) {x->f->init(x);}
    void init(t_fecho_pd *x) {
        try {
            ReservoirInitialiser<FECHOTYPE> netInit;
            netInit.setResRangeLow(resRangeLow).setResRangeHigh(resRangeHigh).setResConnectivity(resConn)
            .setSpectralRadius(alpha)
            .setInConnectivity(inConn).setInRangeLow(inRangeLow).setInRangeHigh(inRangeHigh)
            .init(net);
        } catch (ReservoirInitialiser<FECHOTYPE>::EVException e) {
            post(e.what());
        }
        roInit.setFbConnectivity(fbConn).setFbRangeLow(fbRangeLow).setFbRangeHigh(fbRangeHigh);
        roInit.init(ro);
        inputVec.set_size(x->nIns);
        needToInitialise = false;
        trainInNames.resize(x->nIns);
        for(int i=0; i < x->nIns; i++) trainInNames[i] = NULL;
        trainOutNames.resize(x->nOuts);
        for(int i=0; i < x->nOuts; i++) trainOutNames[i] = NULL;
        post("Initialised");
    }


    
    static void * create_new(t_symbol *selector, int argcount, t_atom *argvec)
    {
        t_fecho_pd *x = (t_fecho_pd *)pd_new(fecho_pd_class);
        post("Create: %d", argcount);
        x->nIns = 1;
        x->nRes = 10;
        x->nOuts = 1;
        bool argError = false;
        if (argcount == 3) {
            if (argvec[0].a_type == A_FLOAT && argvec[1].a_type == A_FLOAT && argvec[2].a_type == A_FLOAT) {
                x->nIns = static_cast<t_int>(argvec[0].a_w.w_float);
                x->nRes = static_cast<t_int>(argvec[1].a_w.w_float);
                x->nOuts = static_cast<t_int>(argvec[2].a_w.w_float);
            }else{
                argError = true;
            }
        }else{
            if (argcount != 0) {
                argError = true;
            }
        }
        if (argError) {
            pd_error(x, "fecho~ takes three arguments: number of inputs, number of reservoir nodes, number of outputs");
        }
        post("fecho~: initialised, %d inputs, %d nodes, %d outputs", x->nIns, x->nRes, x->nOuts);
        x->f = new fecho_pd(x);
        x->out0 = outlet_new(&x->x_obj, &s_list);
        x->out1 = outlet_new(&x->x_obj, &s_list);
        x->out2 = outlet_new(&x->x_obj, &s_float);
        
        x->activationslist = (t_atom *)t_getbytes(sizeof(t_atom) * x->nRes);
        for(int i=0; i < x->nRes; i++) {
            x->activationslist[i].a_type = A_FLOAT;
        }
        x->outputslist = (t_atom *)t_getbytes(sizeof(t_atom) * x->nOuts);
        for(int i=0; i < x->nOuts; i++) {
            x->outputslist[i].a_type = A_FLOAT;
        }
        x->errorslist = (t_atom *)t_getbytes(sizeof(t_atom) * x->nOuts);
        for(int i=0; i < x->nOuts; i++) {
            x->errorslist[i].a_type = A_FLOAT;
        }
        
        return (void *)x;
    }

    static void free(t_fecho_pd *x)
    {
        if (x->f)
            delete x->f;
        if (x->activationslist)
            t_freebytes(x->activationslist, sizeof(t_atom *) * x->nRes);
        if (x->outputslist)
            t_freebytes(x->outputslist, sizeof(t_atom *) * x->nOuts);
    }
};



extern "C" __attribute__((visibility("default"))) void fecho_setup() {
    
    fecho_pd_class = class_new(gensym("fecho"),
                               (t_newmethod)fecho_pd::create_new,
                               (t_method)fecho_pd::free, sizeof(fecho_pd::t_fecho_pd),
                               CLASS_DEFAULT, A_GIMME, A_NULL);
    
    class_addbang(fecho_pd_class, fecho_pd::bang_proxy);
    class_addlist(fecho_pd_class, fecho_pd::listInput_proxy);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::dumpActs_proxy, gensym("getActivations"), A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::reset_proxy, gensym("reset"), A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::randomise_proxy, gensym("randomise"), A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::spectralRadius_proxy, gensym("spectralRadius"), A_FLOAT, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::washout_proxy, gensym("washout"), A_FLOAT, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::resScale_proxy, gensym("resScale"), A_FLOAT, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::resparams_proxy, gensym("resParams"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::inputparams_proxy, gensym("inputParams"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::fbparams_proxy, gensym("fbParams"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::init_proxy, gensym("initialise"), A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::trainInBuffer_proxy, gensym("trainInBuffer"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::trainOutBuffer_proxy, gensym("trainOutBuffer"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::train_proxy, gensym("train"), A_GIMME, A_NULL);
    class_addmethod(fecho_pd_class, (t_method)fecho_pd::trainType_proxy, gensym("trainType"), A_SYMBOL, A_NULL);
}

