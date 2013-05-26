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

void nonLinearFunctionTest() {
    clock_t ts = clock();
    
    ActivationFunctionTanh<FECHOTYPE> resAct;
    Reservoir<FECHOTYPE> net(1, 150, &resAct);
    //    net.setNoise(1e-6);
    
    MultiStreamRecorder<FECHOTYPE> actsRecorder;
    actsRecorder.setChannels(net.getNRes());
    
    ActivationFunctionLinear<FECHOTYPE> roAct;
    ReadOut<FECHOTYPE> ro(net, 1, &roAct);
    ro.setMapInsToOuts(true);
    
    try {
        Initialiser<FECHOTYPE> netInit;
        netInit.setResRangeLow(-1.0).setResRangeHigh(1.0).setResConnectivity(0.2)
        .setSpectralRadius(0.99)
        .setInConnectivity(1.0).setInRangeLow(-1.0).setInRangeHigh(1.0)
        .setFbConnectivity(0.2).setFbRangeLow(-1).setFbRangeHigh(1)
        .init(net, ro);
        net.dump();
//        ro.dump();
    } catch (Initialiser<FECHOTYPE>::EVException e) {
        cout << e.what() << endl;
    }
    Simulator<FECHOTYPE> sim(net, ro);
    
    cout << "Training ESN";
    
    int trainSize=400, testSize = 100;
    Mat<FECHOTYPE> dataIn, dataOut;
    dataIn.set_size(trainSize + testSize, 1);
    dataOut.set_size(trainSize + testSize, 1);
    dataOut(0) = 0;
    for(int i=1; i < trainSize + testSize; i++) {
        dataIn(i) = sin(i/5.0) + (sin(i/7.0)) + cos(i/1.3);
        dataOut(i) = (0.1 * dataIn(i)) + (dataOut(i-1)); //pow(dataIn(i) / 3.0, 2);
    }
    
    Mat<FECHOTYPE> trainIn = dataIn.rows(0, trainSize-1);
    Mat<FECHOTYPE> trainOut = dataOut.rows(0, trainSize-1);
//    TrainerLeastSquares<FECHOTYPE> trainer(&sim, &ro, trainIn, trainOut, 100);
    TrainerPseudoInverse<FECHOTYPE> trainer(&sim, &ro, trainIn, trainOut, 100);
    trainer.train();
    
    cout << "Running ESN\n";
    
    Mat<FECHOTYPE> simIn = dataIn.rows(trainSize, dataIn.n_rows-1);
    Mat<FECHOTYPE> testOut = dataOut.rows(trainSize, dataIn.n_rows-1);
    simIn = trans(simIn);
    Mat<FECHOTYPE> simOut(ro.getSize(), simIn.n_cols);

    
    for(int i=0; i < simIn.n_cols; i++) {
        Col<FECHOTYPE> input=simIn.unsafe_col(i);
        sim.simulate(input);
        Col<FECHOTYPE> res = ro.getOutputs();
        simOut.col(i) = res;
        actsRecorder.addFrame(net.getActivations());
    }

    cout << "processed in " << ((clock() - ts) / (double) CLOCKS_PER_SEC) << " secs\n";
    pyCode << toPyCode("simIn", simIn);
    pyCode << toPyCode("simOut", simOut);
    testOut = trans(testOut);
    pyCode << toPyCode("testOut", testOut);
    pyCode << actsRecorder.dumpToPyCode();
    
}

void sineWaveTest() {
    clock_t ts = clock();
    
    ActivationFunctionTanh<FECHOTYPE> resAct;
    Reservoir<FECHOTYPE> net(0, 20, &resAct);
    net.setNoise(1e-6);
    
    MultiStreamRecorder<FECHOTYPE> actsRecorder;
    actsRecorder.setChannels(net.getNRes());
    
    ActivationFunctionLinear<FECHOTYPE> roAct;
    ReadOut<FECHOTYPE> ro(net, 1, &roAct);
    ro.setMapInsToOuts(false);
    
    try {
        Initialiser<FECHOTYPE> netInit;
        netInit.setResRangeLow(-1.0).setResRangeHigh(1.0).setResConnectivity(0.2)
        .setSpectralRadius(0.45)
        .setInConnectivity(0).setInRangeLow(0).setInRangeHigh(0)
        .setFbConnectivity(1.0).setFbRangeLow(-1).setFbRangeHigh(1)
        .init(net, ro);
        net.dump();
        //        ro.dump();
    } catch (Initialiser<FECHOTYPE>::EVException e) {
        cout << e.what() << endl;
    }
    Simulator<FECHOTYPE> sim(net, ro);
    
    cout << "Training ESN";
    
    int trainSize=4000, testSize = 8000;
    Mat<FECHOTYPE> dataIn, dataOut;
    dataIn.set_size(trainSize + testSize, 1);
    dataOut.set_size(trainSize + testSize, 1);
    dataOut(0) = 0;
    for(int i=1; i < trainSize + testSize; i++) {
        dataIn(i) = 0;
        dataOut(i) = sin(i/100.0) ;
    }
    
    Mat<FECHOTYPE> trainIn = dataIn.rows(0, trainSize-1);
    Mat<FECHOTYPE> trainOut = dataOut.rows(0, trainSize-1);
    //    TrainerLeastSquares<FECHOTYPE> trainer(&sim, &ro, trainIn, trainOut, 100);
    TrainerPseudoInverse<FECHOTYPE> trainer(&sim, &ro, trainIn, trainOut, 2000);
    trainer.train();
    
    cout << "Running ESN\n";
//    net.resetStates();
    
    Mat<FECHOTYPE> simIn = dataIn.rows(trainSize, dataIn.n_rows-1);
    Mat<FECHOTYPE> testOut = dataOut.rows(trainSize, dataIn.n_rows-1);
    simIn = trans(simIn);
    Mat<FECHOTYPE> simOut(ro.getSize(), simIn.n_cols);
    
    
    for(int i=0; i < simIn.n_cols; i++) {
        Col<FECHOTYPE> input=simIn.unsafe_col(i);
        sim.simulate(input);
        Col<FECHOTYPE> res = ro.getOutputs();
        simOut.col(i) = res;
        actsRecorder.addFrame(net.getActivations());
    }
    
    cout << "processed in " << ((clock() - ts) / (double) CLOCKS_PER_SEC) << " secs\n";
    pyCode << toPyCode("simIn", simIn);
    pyCode << toPyCode("simOut", simOut);
    testOut = trans(testOut);
    pyCode << toPyCode("testOut", testOut);
    pyCode << actsRecorder.dumpToPyCode();
    
}


int main(int argc, const char * argv[])
{
    srand((unsigned int)time(NULL));

    std::cout << "Fecho example\n";

    pyCode.open("/tmp/esn.py", ios_base::out);
//    randomNetworkTest();
//    nonLinearFunctionTest();
    sineWaveTest();
    pyCode.close();

    return 0;
}

