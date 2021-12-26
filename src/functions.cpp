#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>
#include <tuple>

#include "constants.hpp"

auto doStuff(const double * dataVec, const int & vectorSize, const int & repeats) -> std::vector<double> & {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    /** We pepper in a few static vars here and there. */
    
    static int i = 0;
    static int j = 0;
    static int q = 0;
    static int t = 0;
    static int repeat = 0;
    
    static double studentT;
    static double groupOneMean = 0;
    static double groupTwoMean = 0;
    static double groupOneStd = 0;
    static double groupTwoStd = 0;
    
    static double discriminationStT;
    static double discriminationStd;
    static double discriminationMean;
    
    static int groupOneCount = 0;
    static int groupTwoCount = 0;
    
    /** We alias some variables so the code looks a touch nicer. */    
    auto & testGroupSize = groupOneCount;
    auto & controlGroupSize = groupTwoCount;
    
    auto & testGroupStd = groupOneStd;
    auto & controlGroupStd = groupTwoStd;
    
    auto & testGroupMean = groupOneMean;
    auto & controlGroupMean = groupTwoMean;
    
    static std::vector <double> holder;
    static std::vector <double> results;
    static std::vector <double> studentTVector;
    results.clear();
    
    
    while(true) {
        
        if (i > vectorSize)
            break;
        
        /** We calculate the mean and keep track of the size of each group. */
        if (dataVec[i] != EMPTY_CELL_VALUE)
            if (i % NUM_COLS < RENAL_SIZE) {
                groupOneMean += dataVec[i];
                groupOneCount ++;
            } else {
                groupTwoMean += dataVec[i];
                groupTwoCount ++;
            }
            
        ++ i;
        
        /** Every NUM_COLS iteration, re-run the last NUM_COLS iterations.
        *    This time we're calculating the sample StdDev using the mean.*/
        if (i % NUM_COLS == 0) {
            j = 0;
            groupOneMean /= groupOneCount;
            groupTwoMean /= groupTwoCount;
            for (q = (i - NUM_COLS); q < i; ++ q) {
                if (dataVec[q] != EMPTY_CELL_VALUE) {
                    if (j < RENAL_SIZE)
                        groupOneStd += ((dataVec[q] - groupOneMean) * (dataVec[q] - groupOneMean));
                    else
                        groupTwoStd += ((dataVec[q] - groupTwoMean) * (dataVec[q] - groupTwoMean));
                    holder.push_back(dataVec[q]);
                }
                j ++;
            }
            
            /** We're taking the sample standard deviation for both groups here. */
            
            groupOneStd /= (groupOneCount - 1);
            groupTwoStd /= (groupTwoCount - 1);
            groupOneStd = std::sqrt(groupOneStd);
            groupTwoStd = std::sqrt(groupTwoStd);
            
            
            studentT = 
                    ((groupOneMean - groupTwoMean) / 
                std::sqrt((
                    ((groupOneStd * groupOneStd) / groupOneCount) + 
                    ((groupTwoStd * groupTwoStd) / groupTwoCount))));
            
            
            discriminationMean = 0;
            discriminationStd = 0;
            
            /** We calculate repeat number of StudentT scores to produce a
            *    discrimination scores. Most of these variables are aliases,
            *    re-using them can incorrectly can lead to unintended behavior.
            */
            for (repeat = 0; repeat < repeats; ++ repeat) {
                std::shuffle(holder.begin(), holder.end(), gen);
                
                testGroupStd = 0;
                controlGroupStd = 0;
                
                testGroupMean = 0;
                controlGroupMean = 0;
                
                for (t = 0; t < holder.size(); ++ t)
                    if (t < testGroupSize)
                        testGroupMean += holder.at(t);
                    else
                        controlGroupMean += holder.at(t);
                
                testGroupMean /= testGroupSize;
                controlGroupMean /= controlGroupSize;
                
                for (t = 0; t < holder.size(); ++ t)
                    if (t < testGroupSize)
                        testGroupStd += ((holder.at(t) - testGroupMean) * (holder.at(t) - testGroupMean));
                    else
                        controlGroupStd += ((holder.at(t) - controlGroupMean) * (holder.at(t) - controlGroupMean));
                
                testGroupStd /= (testGroupSize - 1);
                controlGroupStd /= (controlGroupSize - 1);
                testGroupStd = std::sqrt(testGroupStd);
                controlGroupStd = std::sqrt(controlGroupStd);
                
                discriminationStT = 
                        ((testGroupMean - controlGroupMean) / 
                    std::sqrt((
                        ((testGroupStd * testGroupStd) / testGroupSize) + 
                        ((controlGroupStd * controlGroupStd) / controlGroupSize))));
                        
                studentTVector.push_back(discriminationStT);
                discriminationMean += discriminationStT;
                
            }
            
            discriminationMean /= repeats;
            
            /** Calculate the std deviation of the StudentT distribution. */
            for (auto & it : studentTVector) 
                discriminationStd += ((it - discriminationMean) * (it - discriminationMean));
            
            discriminationStd /= (repeats - 1);
            
            results.push_back(std::abs(studentT - discriminationMean) / std::sqrt(discriminationStd));
            
            groupOneMean = 0;
            groupTwoMean = 0;
            
            groupOneStd = 0;
            groupTwoStd = 0;
            
            groupOneCount = 0;
            groupTwoCount = 0;
            
            holder.clear();
            studentTVector.clear();
        }
    }
    return results;
}

auto studentT(const double * dataVec, const int & vectorSize) -> std::vector<double> & {
    
    /** This is largely the same code from the doStuff function. We split it up 
    *    so that we can use allgather to produce an interesting set of results.
    **
    *    In an ideal world, I would have made the aforementioned code more modular
    *    and simply reused it, but this is not an ideal world.
    */
    
    
    /** We create a couple static variables. */
    static int i = 0;
    static int j = 0;
    static int q = 0;
    
    static double groupOneMean = 0;
    static double groupTwoMean = 0;
    static double groupOneStd = 0;
    static double groupTwoStd = 0;
    
    static int groupOneCount = 0;
    static int groupTwoCount = 0;
    
    static std::vector<double> studentT;
    
    while(true) {
        
        if (i > vectorSize)
            break;
        
        /** We calculate the mean and keep track of the size of each group. */
        if (dataVec[i] != EMPTY_CELL_VALUE)
            if (i % NUM_COLS < RENAL_SIZE) {
                groupOneMean += dataVec[i];
                groupOneCount ++;
            } else {
                groupTwoMean += dataVec[i];
                groupTwoCount ++;
            }
            
        ++ i;
        
        /** Every NUM_COLS iteration, re-run the last NUM_COLS iterations.
        *    This time we're calculating the sample StdDev using the mean.*/
        if (i % NUM_COLS == 0) {
            j = 0;
            groupOneMean /= groupOneCount;
            groupTwoMean /= groupTwoCount;
            for (q = (i - NUM_COLS); q < i; ++ q) {
                if (dataVec[q] != EMPTY_CELL_VALUE)
                    if (j < RENAL_SIZE)
                        groupOneStd += ((dataVec[q] - groupOneMean) * (dataVec[q] - groupOneMean));
                    else
                        groupTwoStd += ((dataVec[q] - groupTwoMean) * (dataVec[q] - groupTwoMean));
                j ++;
            }
            
            /** We're taking the sample standard deviation for both groups here. */
            
            groupOneStd /= (groupOneCount - 1);
            groupTwoStd /= (groupTwoCount - 1);
            groupOneStd = std::sqrt(groupOneStd);
            groupTwoStd = std::sqrt(groupTwoStd);
            
            
            studentT.push_back (((groupOneMean - groupTwoMean) / 
                            std::sqrt((
                                ((groupOneStd * groupOneStd) / groupOneCount) + 
                                ((groupTwoStd * groupTwoStd) / groupTwoCount)))));
                                
            /** Reset these variables within the if statement. */    
            groupOneMean = 0;
            groupTwoMean = 0;
            groupOneStd = 0;
            groupTwoStd = 0;
            
            groupOneCount = 0;
            groupTwoCount = 0;
        }
        
        
        
    }
    return studentT;
}

auto calculateTheD(const double * dataVec, const double * studentT, const int & vectorSize, const int & repeats) -> std::vector<double> & {
    
    /** Splitting this function in this manner requires recalculating some things.
    *    I am doubtful that this will be as efficient as the scatter or broadcast 
    *    methods that were made, but likely not by much.
    */
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    static std::vector <double> holder;
    static std::vector <double> results;
    static std::vector <double> studentTVector;
    
    static int testGroupSize = 0;
    static int controlGroupSize = 0;
    
    static double testGroupMean = 0;
    static double controlGroupMean = 0;
    
    static double testGroupStd = 0;
    static double controlGroupStd = 0;
    
    static double discriminationStT;
    static double discriminationStd;
    static double discriminationMean;
    
    static int i;
    static int j;
    static int t;
    static int repeat;
    
    for (i = 0; i < vectorSize; ++ i) {
        holder.clear();
        
        /** Calculate the number of elements in the test and control groups.
        *    We place all of these items in the holder vector, ignoring 
        *    EMPTY_CELL_VALUE.
        */
        for (j = i * NUM_COLS; j < (i + 1) * NUM_COLS; ++ j)
            if (dataVec[j] != EMPTY_CELL_VALUE) {
                holder.push_back(dataVec[j]);
                if (j % NUM_COLS < RENAL_SIZE)
                    ++ testGroupSize;
                else
                    ++ controlGroupSize;
            }
        
        discriminationMean = 0;
        discriminationStd = 0;
        
        /** We calculate repeat number of StudentT scores to produce a
        *    discrimination scores.
        */
        for (repeat = 0; repeat < repeats; ++ repeat) {
            std::shuffle(holder.begin(), holder.end(), gen);
            
            testGroupStd = 0;
            controlGroupStd = 0;
            
            testGroupMean = 0;
            controlGroupMean = 0;
            
            for (t = 0; t < holder.size(); ++ t)
                if (t < testGroupSize)
                    testGroupMean += holder.at(t);
                else
                    controlGroupMean += holder.at(t);
            
            testGroupMean /= testGroupSize;
            controlGroupMean /= controlGroupSize;
            
            for (t = 0; t < holder.size(); ++ t)
                if (t < testGroupSize)
                    testGroupStd += ((holder.at(t) - testGroupMean) * 
                                     (holder.at(t) - testGroupMean));
                else
                    controlGroupStd += ((holder.at(t) - controlGroupMean) * 
                                        (holder.at(t) - controlGroupMean));
            
            testGroupStd /= (testGroupSize - 1);
            controlGroupStd /= (controlGroupSize - 1);
            testGroupStd = std::sqrt(testGroupStd);
            controlGroupStd = std::sqrt(controlGroupStd);
            
            discriminationStT = 
                        ((testGroupMean - controlGroupMean) / 
                    std::sqrt((
                        ((testGroupStd * testGroupStd) / testGroupSize) + 
                        ((controlGroupStd * controlGroupStd) / controlGroupSize))));
            
            studentTVector.push_back(discriminationStT);
            discriminationMean += discriminationStT;
            
        }
        
        discriminationMean /= repeats;
        
        /** Calculate the std deviation of the StudentT distribution. */
        for (auto & it : studentTVector) 
            discriminationStd += ((it - discriminationMean) * (it - discriminationMean));
        
        discriminationStd /= (repeats - 1);
        results.push_back(std::abs(studentT[i] - discriminationMean) / std::sqrt(discriminationStd));
        
        testGroupSize = 0;
        controlGroupSize = 0;
        
        studentTVector.clear();
        holder.clear();
    }
    
    return results;
}

auto loadDataVector(std::string & file) -> std::vector<double> & {
    
    std::fstream fileIn;
    static std::vector<double> dataVec;
    fileIn.open(file);
    
    /** We store each line in line, ignore first line. */
    std::string line;
    std::getline(fileIn, line);
    
    while (!fileIn.eof()) {
        std::getline(fileIn, line);
        
        /** 
        *    This parsing code is 'borrowed' from my first CIS 677 project.
        */
        std::istringstream stream (line);
        getline(stream, line, ',');
        
        /** Toss the items in a vector, we grab a reference to its pointer in main. */
        while (stream) {
            std::string temp;
            if (!getline(stream, temp, ','))
                break;
            dataVec.push_back(std::stod(temp));
        }        
    }
    return dataVec;
}