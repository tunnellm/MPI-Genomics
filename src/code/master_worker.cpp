#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "../constants.hpp"
#include "../functions.hpp"

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
    int divisor = total_nodes - 1;
    int size_data = (NUM_ROWS / divisor) * NUM_COLS;
    int leftover = (NUM_ROWS * NUM_COLS) - (divisor * size_data);
    
    if (total_nodes == 1) {
        std::cout << "Requires more than just master thread." << std::endl;
        MPI_Finalize();
        exit(0);
    }
    
    if (process_rank == ROOT) {
        
        MPI_Status status;
        
        /** Read the data into a vector. */
        std::string inputString = "NCI-60_processed_v2.csv";
        std::vector <double> dataVec = loadDataVector(inputString);
        
        std::vector <std::vector <double>> outputVector(divisor);
        
        double start = MPI_Wtime();
        
        /** This array is the output array, which is 1/NUM_COLS the
        *    size of the data we sent
        */
        double tempArray[(size_data + leftover) / NUM_COLS];
        
        /** Send the data to the daemons. Each portion of MPI_Send 
        *    on own line for the sake of clarity since two lines are
        *    rather verbose.
        */
        for (int destination = 1; destination < total_nodes; ++destination)
            MPI_Send(
                &dataVec[size_data * (destination - 1)], 
                size_data + (destination == divisor ? leftover : 0), 
                MPI_DOUBLE, 
                destination, 
                DEFAULT_TAG, 
                MPI_COMM_WORLD
            );
        
        for (int i = 1; i < total_nodes; ++i) {
            
            /** We receive more data than needed in all but the last process,
            *    but we will discard the erroneous data.
            */
            MPI_Recv(
                &tempArray, 
                (size_data + leftover) / NUM_COLS, 
                MPI_DOUBLE, 
                MPI_ANY_SOURCE,
                DEFAULT_TAG,
                MPI_COMM_WORLD,
                &status
            ); 
            
            /** We cast from an array to a vector. This may act as 
            *    memcpy behind the scenes. Handles differing sizes for
            *    the leftover data.
            */
            std::vector<double> tempVector (tempArray, tempArray + (size_data + (status.MPI_SOURCE == divisor ? leftover : 0)) / NUM_COLS);
            
            /** Set each inner vector inside of a vector of vectors.
            *    We consider this to be finished once this is done
            *    because reading/writing from this is as trivial as in
            *    the sequential version.
            */            
            outputVector.at(status.MPI_SOURCE - 1) = tempVector;
        }
        
        double end = MPI_Wtime();
        
        /** We output the data here to std out. Use cat to pipe to file. */
        // for (auto & outer : outputVector) {
            // for (auto & inner : outer) {
                // std::cout << inner << std::endl;
            // }
        // }
        std::cout << end - start << std::endl;
        
    } else {
        /** Ensure there's enough room to store the data. */
        double dataVec[size_data + leftover];
        MPI_Recv(
            &dataVec,
            size_data + leftover, 
            MPI_DOUBLE,
            MPI_ANY_SOURCE,
            DEFAULT_TAG,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        
        /** Do stuff to the data and receive the D score. */
        std::vector <double> results = doStuff(
                                        dataVec,
                                        size_data + (process_rank == divisor ? leftover : 0),
                                        num_repeats
                                    );
        /** Make sure we're not sending more data than intended through. */
        MPI_Send (
                &results[0],
                (size_data + (process_rank == divisor ? leftover : 0)) / NUM_COLS,
                MPI_DOUBLE,
                ROOT,
                DEFAULT_TAG,
                MPI_COMM_WORLD
        );
    }
    
    /** Lets clean up here. */
    MPI_Finalize();
    
}









