# MPI-Genomics
 MPI Accelerated Genomics: Genome-wide Association Study
 
Five implementations of an MPI-accelerated genome discrimination algorithm.

This genome discrimination algorithm seeks to determine the true expression of a gene in a test group (renal cancer in this case). This is accomplished by first calculating the Student T score for the expression of a given gene, _G_, in the test group Vs. a control group. We then create a new random permutation of both groups, producing a new "test" group of equal size to the original test group. The Student T score for _G_ is then calculated with this new "test" group against the newly formed control. This is repeated a user-defined number of times, creating a distribution of Student T scores.

Using this new distribution, the likelihood of a highly expressed  _G_ may be determined, providing us with a discrimination score. In essence, this discrimination score allows us to model gene expression in the test group against gene expression occurring at random. A higher discrimination score equates to a lower likelihood of a high gene expression occurring by random chance.

  
