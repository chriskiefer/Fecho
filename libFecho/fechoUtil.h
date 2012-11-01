//
//  fechoUtil.h
//  libFecho
//
//  Created by Chris Kiefer on 01/11/2012.
//  Copyright (c) 2012 Goldsmiths, University of London. EAVI. All rights reserved.
//

#ifndef libFecho_fechoUtil_h
#define libFecho_fechoUtil_h

#include <sstream>
#include <iostream>

template <typename T>
class MultiStreamRecorder {
public:
    void setChannels(int num) {
        streams.resize(num);
        clear();
    }
    void addFrame(vector<T> &vals) {
        for(int i=0; i < streams.size(); i++) {
            streams[i].push_back(vals[i]);
        }
    }
    void clear() {
        for(int i=0; i < streams.size(); i++) {
            streams[i].clear();
        }
    };
    
    vector<vector<T> > streams;
    
    string dumpToPyCode() {
        stringstream s;
        s << "x = array([";
        for(int i=0; i < streams.size(); i++) {
            s << "[";
            for(int j=0; j < streams[i].size(); j++) {
                s << streams[i][j] << ",";
            }
            s << "],";
        }
        s << "]);\n";
        return s.str();
    }
};

template<typename T>
class MSE {
    static T calc(vector<T> &seq1, vector<T> &seq2) {
        
    }
};


#endif
