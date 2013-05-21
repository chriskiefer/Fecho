//
//  main.cpp
//  FechoExample
//
//  Created by Chris on 13/05/2013.
//  Copyright (c) 2013 Chris Kiefer. All rights reserved.
//

#include <iostream>
#include "fecho.hpp"
#include <Accelerate/Accelerate.h>
#include <fstream>
#include "armadillo"

using namespace Fecho;
ofstream pyCode;
typedef double FECHOTYPE;

void randomNetworkTest() {
    //run a network with random weights, ignore the readout and observe the interal activations
    clock_t ts = clock();
    
    ActivationFunctionTanh<FECHOTYPE> resAct;
    Reservoir<FECHOTYPE> net(0, 10, &resAct);
//    net.setNoise(1e-6);
    
    MultiStreamRecorder<FECHOTYPE> actsRecorder;
    actsRecorder.setChannels(net.getNRes());
    
    ActivationFunctionLinear<FECHOTYPE> roAct;
    ReadOut<FECHOTYPE> ro(net, 1, &roAct);
    ro.setMapInsToOuts(true);
    
    try {
        Initialiser<FECHOTYPE> netInit;
        netInit.setResRangeLow(-0.1).setResRangeHigh(0.1).setResConnectivity(0.5)
        .setSpectralRadius(1.1)
        .setInConnectivity(0.0).setInRangeLow(0).setInRangeHigh(0)
        .setFbConnectivity(0.0).setFbRangeLow(-1).setFbRangeHigh(1)
        .init(net, ro);
        net.dump();
        ro.dump();
    } catch (Initialiser<FECHOTYPE>::EVException e) {
        cout << e.what() << endl;
    }
    Simulator<FECHOTYPE> sim(net, ro);
    
    int runSize = 500;
    cout << "Running ESN\n";
    Col<FECHOTYPE> inputVec;
    inputVec.set_size(0);
//    Col<FECHOTYPE> res(runSize);
    //net.setNoise(0);
    // net.resetStates();
//    net.randomiseStates();
    Col<FECHOTYPE> initStates;
    initStates.set_size(net.getNRes());
    initStates.fill(1.0);
    for(int i=0; i < runSize; i++) {
        if (i % 50 == 0) {
            net.setStates(initStates);
        }
        sim.simulate(inputVec);
        actsRecorder.addFrame(net.getActivations());
    }
    
    cout << "processed in " << ((clock() - ts) / (double) CLOCKS_PER_SEC) << " secs\n";
    pyCode << actsRecorder.dumpToPyCode();
    
}

void sineWaveTest() {
    clock_t ts = clock();
    
    ActivationFunctionTanh<FECHOTYPE> resAct;
    Reservoir<FECHOTYPE> net(0, 4, &resAct);
    net.setNoise(1e-6);
//    
//    MultiStreamRecorder<prec> actsRecorder;
//    actsRecorder.setChannels(net.getNRes());
//    
    ActivationFunctionLinear<FECHOTYPE> roAct;
    ReadOut<FECHOTYPE> ro(net, 1, &roAct);
    ro.setMapInsToOuts(true);
    
    try {
        Initialiser<FECHOTYPE> netInit;
        netInit.setResRangeLow(-0.1).setResRangeHigh(0.1).setResConnectivity(0.5)
        .setSpectralRadius(0.1)
        .setInConnectivity(0.0).setInRangeLow(0).setInRangeHigh(0)
        .setFbConnectivity(1.0).setFbRangeLow(-1).setFbRangeHigh(1)
        .init(net, ro);
        net.dump();
        ro.dump();
    } catch (Initialiser<FECHOTYPE>::EVException e) {
        cout << e.what() << endl;
    }
    
    

//    //    SimulatorLI<prec> sim(net, ro, 0.9);
    Simulator<FECHOTYPE> sim(net, ro);
    
//    vector<FECHOTYPE> ins;
//    ins.push_back(0);
//    
//    Col<FECHOTYPE> trainIn, trainOut;
//    int trainSize = 300;
    int runSize = 100;
//    trainIn.resize(0);
//    trainOut.resize(trainSize);
//    
//    vector<FECHOTYPE> sig(trainSize + runSize);
//    for(int i=0; i < sig.size(); i++) {
//        FECHOTYPE r = (0.5 *sin(i/4.0));
//        sig[i] = r;
//    }
//    
//    std::copy(sig.begin(), sig.begin() + trainSize, trainOut.begin());
//    
//    vector<FECHOTYPE> testOut(runSize);
//    std::copy(sig.begin() + trainSize, sig.end(), testOut.begin());
//
//    TrainerPseudoInverse<prec> tr(&sim, &ro, trainIn, trainOut, trainSize, 100);
//    cout << "Training...\n";
//    tr.train();
//    
//    cout << "Running ESN\n";
//    Col<FECHOTYPE> res(runSize);
//    //net.setNoise(0);
//    // net.resetStates();
//    for(int i=0; i < runSize; i++) {
//        sim.simulate(&trainIn[0]);
//        res[i] = ro.getOutputs()[0];
//        actsRecorder.addFrame(net.getActivations());
//    }
    
    cout << "processed in " << ((clock() - ts) / (double) CLOCKS_PER_SEC) << " secs\n";
//    
//    pyCode << "simIn = array([";
//    //    for(int i=0; i < runSize; i++) {
//    //        cout << trainIn[i] << ",";
//    //        pyCode << trainIn[i] << ",";
//    //    }
//    //    cout << endl;
//    pyCode << "])\n";
//    
//    cout << "Result: \n";
//    pyCode << "simOut = array([";
//    for(int i=0; i < runSize; i++) {
//        cout << res[i] << ",";
//        pyCode << res[i] << ",";
//    }
//    pyCode << "])\n";
//    cout << endl;
//    
//    
//    pyCode << toPyCode("testOut", testOut);
//    
//    pyCode << actsRecorder.dumpToPyCode();
//    
//    cout << "MSE: " << MSE<prec>::calc(res, testOut) << endl;
    
}


int main(int argc, const char * argv[])
{
    srand((unsigned int)time(NULL));

    std::cout << "Fecho example\n";

    pyCode.open("/tmp/esn.py", ios_base::out);
    randomNetworkTest();
//    sineWaveTest();
    pyCode.close();

    return 0;
}

