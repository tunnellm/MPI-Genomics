#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>

#include "../functions.hpp"


auto main (int argc, char ** argv) -> int {
    
    static int repeats;
    
    repeats = std::stoi(argv[1]);
    
    std::string inputString = "NCI-60_processed_v2.csv";
    std::vector<double> dataVec = loadDataVector(inputString);
    
    auto startTime = std::chrono::steady_clock::now();
    doStuff(&dataVec[0], 4540, repeats);
    auto endTime = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> elapsed_seconds = endTime-startTime;
    
    std::cout << elapsed_seconds.count() << std::endl;
    return 1;
    
}