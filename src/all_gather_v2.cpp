#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <iterator>
#include <numeric>

#include "constants.hpp"
#include "functions.hpp"

#include <mpi.h>

auto main (int argc, char ** argv) -> int {
    
    static int num_repeats;
    
    num_repeats = std::stoi(argv[1]);
    
    static int process_rank;
    static int total_nodes;
    
    /** Initializer code*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    
    /** Calculate the amount of data we need to send to each daemon.
    *    It's important to note that we need to break the data wrt
    *    NUM_COLS.
    */
    static int divisor = total_nodes;
    static int size_data = (NUM_ROWS / divisor) * NUM_COLS;
    static int leftover = (NUM_ROWS * NUM_COLS) - (divisor * size_data);
    
    if (total_nodes == 1) {
        std::cout << "Requires more than just master thread." << std::endl;
        MPI_Finalize();
        exit(0);
    }
    
    /** Initialize the arrays */
    static std::vector <double> dataVec(NUM_ROWS * NUM_COLS);
    static std::vector <double> intermediaryVec(size_data + NUM_COLS);
    static std::vector <double> allStudentT(NUM_ROWS, 0);
    static std::vector <double> allD;
    static std::vector <double> outputVec(NUM_ROWS, 0);
    
    if (process_rank == ROOT) {
        
        /** Read the data into a vector. */
        std::string inputString = "NCI-60_processed_v2.csv";
        dataVec = loadDataVector(inputString);
        
    }
    
    /** We start timing now that all of the data is loaded. */
    double start = MPI_Wtime();
    
    /** We calculate how much data to send to each node. While there is leftover
    *    data remaining, nodes are given (size_data + NUM_COLS) data. Afterward,
    *    just NUM_COLS data is dispersed.
    */
    static std::vector <int> dataInputSize;
    static std::vector <int> dataOffset(total_nodes, 0);
    
    for (int i = 0; i < total_nodes; ++ i) {
        if (leftover >= NUM_COLS) {
            dataInputSize.push_back(size_data + NUM_COLS);
            leftover -= NUM_COLS;
        } else {
            dataInputSize.push_back(size_data);
        }
    }
    
    static int sum = 0;
    
    /** This for loop mimics the functionality of std::exclusive_scan--mpic++ does not 
    *    appear to support it. The compiler just says that it is not part of the std 
    *    library despite having the correct imports. This may be due to the fact that 
    *    new implementations would appear to be multithreaded, but I'm not sure.
    */
    for (int i = 1; i < dataInputSize.size(); ++ i)
        dataOffset.at(i) = (sum = sum + dataInputSize.at(i - 1));
    
    /** Broadcast all of the data to all nodes. We'll perform half of the work in parallel,
    *    then scatter it to the other nodes.
    */
    MPI_Bcast(&dataVec[0], dataVec.size(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);    
    
    /** We calculate the studentT scores in parallel, then allgather to pass to other 
    *    nodes. We have access to the entire array in this version, so we pass the 
    *    starting values that we care about. There are minor changes throughout the rest
    *    of this file when compared with the original all_gather, but this is the only
    *    substantial one.
    */    
    intermediaryVec = studentT(&dataVec[dataOffset.at(process_rank)], dataInputSize.at(process_rank));
    
    
    /** We decrease the size of the inputs and offsets to match the output data size,
    *    it's all relative.
    */
    for (int i = 0; i < dataInputSize.size(); ++ i) {
        dataInputSize.at(i) /= NUM_COLS;
        dataOffset.at(i) /= NUM_COLS;
    }

    
    /** https://www.mpich.org/static/docs/v3.1/www3/MPI_Gatherv.html */
    
    /** MPI_Gatherv is similar to MPI_Gather, except it pairs with MPI_Scatter in that it
    *    requires a list of data sizes and corresponding offsets. These data sizes and offsets
    *    will differ from the ones sent with Scatterv should the data sizes change at all.
    **    
    *    For some reason, I was not able to use the same receiving array as the sending array.
    *    I'm not sure if this is intended behavior, but it would not work properly otherwise.
    */
    
    
    /** Now we pass the StudentT scores around to all of the nodes. */
    MPI_Allgatherv(&intermediaryVec[0], dataInputSize.at(process_rank), MPI_DOUBLE, &allStudentT[0], &dataInputSize[0], &dataOffset[0], MPI_DOUBLE, MPI_COMM_WORLD);
    
    /** Calculate the D - Score, repeat 1/total_nodes times, then we take the average. */
    allD = calculateTheD(&dataVec[0], &allStudentT[0], NUM_ROWS, num_repeats / total_nodes);
    
    /** We reduce and bring the D scores together now. It is considered done once this 
    *    returns.
    */    
    MPI_Reduce(&allD[0], &outputVec[0], allD.size(), MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    
    double end = MPI_Wtime();
    
    
    if (process_rank == ROOT) {
        // for (auto & it : outputVec)
            // std::cout << it / total_nodes << std::endl;
        std::cout << end - start << std::endl;
    }
    
    /** Lets clean up here. */
    MPI_Finalize();
    
    return 0;
    
}









