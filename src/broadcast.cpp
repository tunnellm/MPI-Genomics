#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "constants.hpp"
#include "functions.hpp"

#include <mpi.h>

auto main (int argc, char ** argv) -> int {
    
    static int num_repeats;
    
    num_repeats = std::stoi(argv[1]);
    
    int process_rank;
    int total_nodes;
    
    /** Initializer code*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    
    /** Calculate the amount of data we need to send to each daemon.
    *    It's important to note that we need to break the data wrt
    *    NUM_COLS.
    */
    
    if (total_nodes == 1) {
        std::cout << "Requires more than just master thread." << std::endl;
        MPI_Finalize();
        exit(0);
    }
    
    /** Initialize the array */
    static std::vector <double> dataVec(NUM_ROWS * NUM_COLS);
    static std::vector <double> intermediaryVec(NUM_ROWS);
    static std::vector <double> outputVec(NUM_ROWS, 0);
    
    if (process_rank == ROOT) {
        /** Read the data into a vector. */
        std::string inputString = "NCI-60_processed_v2.csv";
        dataVec = loadDataVector(inputString);
    }
    
    double start = MPI_Wtime();
    
    /** Send all of the data to all of the nodes, source is ROOT node.*/
    MPI_Bcast(&dataVec[0], dataVec.size(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    
    /** Do stuff to the data and receive the D score. */
    intermediaryVec = doStuff(&dataVec[0], dataVec.size(), num_repeats / total_nodes);
    
    /** We perform a sum reduction on the data, storing in output. We consider
    *    this done, but will output the averaged values afterward.
    */
    MPI_Reduce(&intermediaryVec[0], &outputVec[0], outputVec.size(), MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    
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









