//
//  dSFMTRand.h
//  Esnesnesn
//
//  Created by Chris Kiefer on 13/09/2012.
//  Copyright (c) 2012 Goldsmiths, University of London. EAVI. All rights reserved.
//

#ifndef Esnesnesn_dSFMTRand_h
#define Esnesnesn_dSFMTRand_h

extern "C" {
    #include "dSFMT/dSFMT.h"
}

class dSFMTRand {
public:
    dsfmt_t *mtRand;
    dSFMTRand() {
        mtRand = new dsfmt_t();
        dsfmt_init_gen_rand(mtRand, (int)time(NULL));  
        dsfmt_gv_init_gen_rand((int)time(NULL));
    }
    ~dSFMTRand() {
        delete mtRand;
    }
    inline double randUF() {
        return dsfmt_genrand_close_open(mtRand);
    }
    inline double randF() {
        return (dsfmt_genrand_close_open(mtRand) - 0.5) * 2.0;
    }
    
    inline unsigned int randUInt() {
        return dsfmt_gv_genrand_uint32();        
    }
    
    inline void randArray(double vals[], int size) {
        dsfmt_gv_fill_array_close_open(vals,size);
    }
};

#endif
