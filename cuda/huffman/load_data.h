#ifndef _LOADTESTDATA_H_
#define _LOADTESTDATA_H_

//#include "testdatagen.h"
#include "hist.cu"
#include "huffTree.h"

inline void initParams(char *file_name, uint num_block_threads, uint &num_blocks, uint &num_elements, uint &mem_size, uint symbol_type_size){
    if (file_name == NULL) {
        num_elements = num_blocks * num_block_threads;
        mem_size = num_elements * symbol_type_size;
    }
    else {
        FILE *f = fopen(file_name, "rb");
        if (!f) { perror(file_name); exit(1); }
        fseek(f, 0, SEEK_END);
        mem_size = ftell(f);
        fclose(f);
        num_elements = mem_size / symbol_type_size;
        //todo add check if we need 1 more block!
        num_blocks = num_elements / num_block_threads;
    }
}

inline void loadData(char *file_name, uint *sourceData, uint *codewords, uint *codewordlens, uint num_elements, uint mem_size, double &H){
    if (file_name == NULL) {
        printf("No input file\n");
        exit(-1);
    }
    else 
      {
        unsigned int freqs[UniqueSymbols] = {0};
        runHisto(file_name,freqs,mem_size,sourceData);
        INode* root = BuildTree(freqs);

        HuffCodeMap codes;
        GenerateCodes(root, HuffCode(), codes);
        delete root;

        for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it)
          {
            unsigned int count = distance(it->second.begin(),it->second.end());
            for(int i = 0; i < count; i++)
              if(it->second[i]) 
                codewords[(unsigned int)(it->first)]+=(uint)pow(2.0f,(int)count - i - 1);
            codewordlens[(unsigned int)(it->first)]=count;
          }

        H = 0.0;
        for (unsigned int i=0; i<256; i++)
          if (freqs[i] > 0) {
              double p = (double)freqs[i] / (double)mem_size;
              H += p * log(p) / log(2.0);
          }
        H = -H;
        printf("\n%s, %u bytes, entropy %f\n\n", file_name, mem_size, H);
      }
}

#endif
